# Pointing executorch's bundled XNNPACK at a working tree

When developing XNNPACK kernels that need to flow through executorch's
delegate path (e.g. for the LeNet bf16 export pipeline), the
executorch-bundled XNNPACK submodule needs to pick up the in-progress
changes from your working tree. The simplest approach is a symlink swap.

## Setup

From the executorch third-party directory:

```sh
cd /users/jonathan.l/mytools/C/zephyr-chipyard-sw/third-party/executorch/backends/xnnpack/third-party

# Preserve the original (do NOT delete — it owns the submodule .git pointer)
mv XNNPACK XNNPACK.orig

# Point at your working tree
ln -s /users/jonathan.l/mytools/C/XNNPACK XNNPACK
```

The `.orig` directory keeps the submodule's `.git` gitlink file intact,
which means git tooling that walks the executorch repo still has a
coherent view. Don't `rm -rf XNNPACK.orig` — restoring it later requires
this gitlink.

## Why `git status` looks identical in both directories

`XNNPACK.orig/.git` is a one-line `gitdir:` pointer, e.g.:

```
gitdir: ../../../../../../.git/modules/third-party/executorch/modules/backends/xnnpack/third-party/XNNPACK
```

That gitdir's `core.worktree` config resolves through the symlinked
`XNNPACK` (which now points at your working tree). So `git -C
XNNPACK.orig status` and `git -C XNNPACK status` both report the working
tree's state — they share the same worktree as far as git is concerned.

To actually inspect the physical files inside `XNNPACK.orig` rather than
following the symlink, override both env vars explicitly:

```sh
GIT_DIR=XNNPACK.orig/.git GIT_WORK_TREE=XNNPACK.orig git status
```

…but in practice this is rarely useful — once the swap is in place, all
work happens in the symlinked working tree.

## Mirroring local-only edits into the working tree

If the bundled XNNPACK had uncommitted edits before the swap, those live
inside `XNNPACK.orig/` (physically — not via the symlink). To pull them
into your working tree, diff the two directories and replay the relevant
edits manually:

```sh
diff -u XNNPACK.orig/some/file.c /bwrcq/C/jonathan.l/XNNPACK/some/file.c
```

Don't blanket-copy: the working tree may have its own newer changes for
the same files. Replay edit-by-edit.

## Reverting

To detach the symlink and restore the original submodule contents:

```sh
cd /users/jonathan.l/mytools/C/zephyr-chipyard-sw/third-party/executorch/backends/xnnpack/third-party

rm XNNPACK              # remove the symlink (NOT -rf — it's just a symlink)
mv XNNPACK.orig XNNPACK # restore original directory + gitlink in one move
```

Verify the gitlink resolves cleanly:

```sh
git -C XNNPACK status   # should report on the original submodule's HEAD
```

If `XNNPACK.orig` is gone (e.g. accidentally deleted), recover by
re-checking out the submodule from the executorch superproject:

```sh
cd /users/jonathan.l/mytools/C/zephyr-chipyard-sw
git submodule update --init --recursive third-party/executorch/backends/xnnpack/third-party/XNNPACK
```

…but this loses any uncommitted edits that were in the original
directory. Hence: don't delete `.orig` until you've confirmed nothing
inside it is needed.
