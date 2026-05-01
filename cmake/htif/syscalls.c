// Minimal HTIF syscall layer for spike bare-metal targets.
//
// HTIF protocol (riscv-isa-sim spec):
//   - tohost / fromhost are two 8-byte memory cells at fixed locations.
//   - Console output: tohost = ptr_to_magic_buf, where
//     magic_buf[0] = SYS_write (64), magic_buf[1] = fd, magic_buf[2] = buf,
//     magic_buf[3] = nbytes. Spike writes the result to magic_buf[0] and
//     sets fromhost = 1.
//   - Exit: tohost = (code << 1) | 1.
//
// We only need write/exit + a few stubs for picolibc (sbrk, isatty,
// gettimeofday, fstat, etc.) because gtest is built with
// -DGTEST_HAS_FILE_SYSTEM=0 -DGTEST_HAS_PTHREAD=0 etc.

#include <stddef.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>

#define SYS_write 64

// .tohost section places these at fixed addresses (see link.ld).
volatile uint64_t tohost   __attribute__((section(".tohost"), aligned(64))) = 0;
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64))) = 0;

static long htif_syscall(long which, long a0, long a1, long a2) {
  static volatile uint64_t magic_mem[8] __attribute__((aligned(64)));
  magic_mem[0] = (uint64_t) which;
  magic_mem[1] = (uint64_t) a0;
  magic_mem[2] = (uint64_t) a1;
  magic_mem[3] = (uint64_t) a2;
  __sync_synchronize();
  tohost = (uint64_t) (uintptr_t) magic_mem;
  while (fromhost == 0) { /* spin */ }
  fromhost = 0;
  __sync_synchronize();
  return (long) magic_mem[0];
}

// POSIX I/O — write to spike console, others stubbed.
ssize_t write(int fd, const void *buf, size_t count) {
  return htif_syscall(SYS_write, fd, (long)(uintptr_t)buf, (long) count);
}
ssize_t _write(int fd, const void *buf, size_t count) {
  return write(fd, buf, count);
}
ssize_t read(int fd, void *buf, size_t count)        { (void)fd; (void)buf; (void)count; return -1; }
ssize_t _read(int fd, void *buf, size_t count)       { return read(fd, buf, count); }
off_t   lseek(int fd, off_t off, int whence)         { (void)fd; (void)off; (void)whence; return -1; }
off_t   _lseek(int fd, off_t off, int whence)        { return lseek(fd, off, whence); }
int     close(int fd)                                { (void)fd; return 0; }
int     _close(int fd)                               { return close(fd); }
int     open(const char *p, int f, ...)              { (void)p; (void)f; return -1; }
int     _open(const char *p, int f, ...)             { return open(p, f); }
int     isatty(int fd)                               { return fd <= 2; }
int     _isatty(int fd)                              { return isatty(fd); }

// Stub localtime_r / gettimeofday / fstat / kill / getpid for libstdc++
// chrono and gtest fallbacks. They're never called in a way that matters.
struct timeval { long tv_sec; long tv_usec; };
int gettimeofday(struct timeval *tv, void *tz) {
  (void) tz; if (tv) { tv->tv_sec = 0; tv->tv_usec = 0; } return 0;
}
struct stat;
int fstat(int fd, struct stat *st)  { (void)fd; (void)st; return -1; }
int _fstat(int fd, struct stat *st) { return fstat(fd, st); }
int kill(int pid, int sig)          { (void)pid; (void)sig; return -1; }
int _kill(int pid, int sig)         { return kill(pid, sig); }
int getpid(void)                    { return 1; }
int _getpid(void)                   { return getpid(); }

// `_exit` propagates the C exit code through HTIF so spike sees it.
__attribute__((noreturn)) void _exit(int code) {
  tohost = (uint64_t)((code << 1) | 1);
  __sync_synchronize();
  while (1) { /* spin */ }
}
__attribute__((noreturn)) void exit(int code) { _exit(code); }
__attribute__((noreturn)) void abort(void)    { _exit(134); }

// `_sbrk` for picolibc malloc. Heap region is __heap_start..__heap_end
// from the linker script.
extern char __heap_start[];
extern char __heap_end[];
void *_sbrk(ptrdiff_t incr) {
  static char *brk = NULL;
  if (brk == NULL) brk = __heap_start;
  if (brk + incr > __heap_end) return (void *) -1;
  char *prev = brk;
  brk += incr;
  return prev;
}
void *sbrk(ptrdiff_t incr) { return _sbrk(incr); }

// Picolibc tinystdio expects the application to provide stdin/stdout/
// stderr as `FILE *const`. We point all three at one __file struct
// whose `put` callback writes to fd 1 via our HTIF write().
#include <stdio.h>
static int htif_put_callback(char c, FILE *f) {
  (void) f;
  unsigned char ch = (unsigned char) c;
  write(1, &ch, 1);
  return ch;
}
static int htif_get_callback(FILE *f) {
  (void) f;
  return -1;  // EOF; gtest doesn't read stdin in our config.
}
static FILE __htif_file =
    FDEV_SETUP_STREAM(htif_put_callback, htif_get_callback, NULL,
                      _FDEV_SETUP_RW);
FILE *const stdin  = &__htif_file;
FILE *const stdout = &__htif_file;
FILE *const stderr = &__htif_file;
