// Minimal HTIF bare-metal startup for RV64 binaries running under spike.
//
// Memory layout (matches link.ld):
//   .text/.data/.bss : 0x80000000 ..
//   .tohost          : 0x80001000  (tohost @ +0, fromhost @ +0x40)
//   _end             : end of bss
//   __stack_top      : top of stack (defined in linker script)
//   __heap_start     : after BSS, after .stack
//   __heap_end       : __heap_start + heap size
//
// On entry, spike jumps to _start (the ELF entry). We:
//   1. Zero general-purpose regs (good hygiene).
//   2. Enable F + V state in mstatus so the binary can use them.
//   3. Set gp/sp.
//   4. Zero .bss.
//   5. Run __libc_init_array (so C++ static ctors fire) — CRITICAL for gtest.
//   6. Call main(0, NULL).
//   7. Run __libc_fini_array.
//   8. Write (return_code << 1) | 1 to tohost; halt.
//
// On any trap (illegal instr, store fault, etc.), trap_entry writes
// (1337 << 1) | 1 to tohost so spike sees a non-zero exit.

#define MSTATUS_FS_INITIAL  (1 << 13)
#define MSTATUS_VS_INITIAL  (1 << 9)

  .section .text.init,"ax",@progbits
  .globl _start
_start:
  // Zero all GPRs (except x0 which is hardwired and tp/gp set below).
  li  x1,  0
  li  x2,  0
  li  x4,  0
  li  x5,  0
  li  x6,  0
  li  x7,  0
  li  x8,  0
  li  x9,  0
  li  x10, 0
  li  x11, 0
  li  x12, 0
  li  x13, 0
  li  x14, 0
  li  x15, 0
  li  x16, 0
  li  x17, 0
  li  x18, 0
  li  x19, 0
  li  x20, 0
  li  x21, 0
  li  x22, 0
  li  x23, 0
  li  x24, 0
  li  x25, 0
  li  x26, 0
  li  x27, 0
  li  x28, 0
  li  x29, 0
  li  x30, 0
  li  x31, 0

  // Enable FPU + Vector unit (mstatus.FS = Initial, mstatus.VS = Initial).
  li   t0, (MSTATUS_FS_INITIAL | MSTATUS_VS_INITIAL)
  csrs mstatus, t0

  // Set up trap vector → trap_entry (writes 1337 to tohost on any trap).
  la   t0, trap_entry
  csrw mtvec, t0

  // gp = __global_pointer$ (relaxation needs this).
.option push
.option norelax
  la gp, __global_pointer$
.option pop

  // sp = top of stack (linker script).
  la sp, __stack_top

  // tp = TLS base. We reserve a small TLS region in BSS at __tls_base.
  la tp, __tls_base

  // Zero .bss.
  la t0, __bss_start
  la t1, __bss_end
1:
  beq  t0, t1, 2f
  sd   x0, 0(t0)
  addi t0, t0, 8
  j    1b
2:

  // Run global C++ constructors / __attribute__((constructor)) functions.
  call __libc_init_array

  // call main(argc=1, argv={"test", NULL}, envp=NULL).
  // gtest_main's InitGoogleTest needs a non-null argv to mark itself
  // as initialized; otherwise RUN_ALL_TESTS prints a "DO NOT IGNORE"
  // notice. argv lives in .rodata below.
  li  a0, 1
  la  a1, __htif_argv
  li  a2, 0
  call main

  // a0 = main's return code. Run global dtors.
  mv   s0, a0
  call __libc_fini_array
  mv   a0, s0

  // exit(a0) — write (code << 1) | 1 to tohost and spin.
  slli t0, a0, 1
  ori  t0, t0, 1
  la   t1, tohost
  sd   t0, 0(t1)
  fence
3: j 3b

  // argv = { __htif_progname, NULL } — referenced by main() above.
  .section .rodata
  .align 3
__htif_argv:
  .dword __htif_progname
  .dword 0
__htif_progname:
  .asciz "test"

  .section .text.init,"ax",@progbits
  // Trap handler — any unexpected exception writes "fail" exit code.
  .align 2
  .globl trap_entry
trap_entry:
  li   t0, (1337 << 1) | 1
  la   t1, tohost
  sd   t0, 0(t1)
  fence
4: j 4b
