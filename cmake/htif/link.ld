/* Linker script for HTIF bare-metal binaries running on spike.
 *
 * Memory layout:
 *   0x80000000  .text.init (entry _start) + .text + .rodata
 *   0x80001000  .tohost (must be at this fixed addr — spike scans for it)
 *   ...         .data, .sdata, .bss, .sbss
 *   __bss_end   marks the end of bss
 *   __tls_base  small TLS region (just past bss, picolibc uses thread-local)
 *   __stack_top top-of-stack (downward-growing)
 *   __heap_start..__heap_end  heap for picolibc sbrk
 *
 * Sizes are pessimistic but bounded so we don't burn unlimited memory.
 */

OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 256M
}

/* Stack and heap sizes — gtest allocates a lot of strings, give it room. */
__stack_size = 1M;
__heap_size  = 64M;
__tls_size   = 4K;

SECTIONS
{
  . = 0x80000000;

  .text.init : { *(.text.init) } > RAM
  .tohost ALIGN(0x1000) : {
    /* Spike scans the ELF for `tohost` and `fromhost` symbols and uses
     * the addresses it finds — the magic 0x80001000 placement is just
     * convention. Both symbols live in this section.
     */
    *(.tohost)
  } > RAM

  .text : ALIGN(0x40) {
    *(.text .text.* .gnu.linkonce.t.*)
  } > RAM

  .rodata : ALIGN(0x40) {
    *(.rdata .rodata .rodata.* .gnu.linkonce.r.*)
  } > RAM

  /* Picolibc's __libc_init_array iterates from __bothinit_array_start
   * to __bothinit_array_end. Both symbols are weak-referenced from
   * libc.a, so we MUST emit strong definitions (PROVIDE doesn't work
   * for weak refs) to override the default 0. */
  .preinit_array : ALIGN(8) {
    __bothinit_array_start = .;
    __preinit_array_start  = .;
    KEEP(*(.preinit_array))
    __preinit_array_end    = .;
  } > RAM
  .init_array : ALIGN(8) {
    __init_array_start = .;
    KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*) SORT_BY_INIT_PRIORITY(.ctors.*)))
    KEEP(*(.init_array EXCLUDE_FILE(*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o) .ctors))
    __init_array_end     = .;
    __bothinit_array_end = .;
  } > RAM
  .fini_array : ALIGN(8) {
    __fini_array_start = .;
    KEEP(*(SORT_BY_INIT_PRIORITY(.fini_array.*) SORT_BY_INIT_PRIORITY(.dtors.*)))
    KEEP(*(.fini_array EXCLUDE_FILE(*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o) .dtors))
    __fini_array_end = .;
  } > RAM

  /* Exception-unwind tables (libstdc++ throw machinery). */
  .eh_frame_hdr : { *(.eh_frame_hdr) *(.eh_frame_entry .eh_frame_entry.*) } > RAM
  .eh_frame    : ONLY_IF_RO { KEEP (*(.eh_frame)) *(.eh_frame.*) } > RAM
  .gcc_except_table : ONLY_IF_RO { *(.gcc_except_table .gcc_except_table.*) } > RAM
  .gnu_extab : ONLY_IF_RO { *(.gnu_extab*) } > RAM

  .data : ALIGN(0x40) {
    *(.data .data.* .gnu.linkonce.d.*)
    *(.eh_frame .eh_frame.*)
    *(.gcc_except_table .gcc_except_table.*)
  } > RAM

  .sdata : ALIGN(8) {
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.* .sdata2.* .gnu.linkonce.s.*)
  } > RAM

  /* TLS: we don't actually run a multi-threaded environment, but
   * picolibc references __thread variables (errno, locale). Put their
   * tdata image after .sdata, then bss-zero them.
   */
  .tdata : ALIGN(8) {
    PROVIDE(__tdata_start = .);
    *(.tdata .tdata.*)
    PROVIDE(__tdata_end = .);
  } > RAM

  .bss : ALIGN(0x40) {
    PROVIDE(__bss_start = .);
    *(.tbss .tbss.*)
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.bss .bss.* .gnu.linkonce.b.*)
    *(COMMON)
    . = ALIGN(8);
    PROVIDE(__bss_end = .);
  } > RAM

  /* TLS region — picolibc uses a small per-thread block via tp. */
  . = ALIGN(0x40);
  __tls_base = .;
  . = . + __tls_size;

  /* Stack — grows down from __stack_top. */
  . = ALIGN(0x40);
  __stack_bottom = .;
  . = . + __stack_size;
  __stack_top    = .;

  /* Heap — picolibc sbrk uses [__heap_start, __heap_end). */
  . = ALIGN(0x40);
  __heap_start = .;
  . = . + __heap_size;
  __heap_end   = .;

  _end = .;

  /DISCARD/ : { *(.note.GNU-stack) *(.comment) }
}
