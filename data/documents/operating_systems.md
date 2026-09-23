# Operating Systems Basics

An operating system manages a computer's hardware and provides a
common set of services so that programs don't each have to talk to the
hardware directly. Windows, Linux, and macOS are all operating
systems.

## Processes

A process is a running instance of a program, with its own memory
space, open files, and execution state. When you open a web browser
and a text editor at the same time, each is running as a separate
process, and one crashing normally does not bring down the other. The
operating system keeps track of every process and decides which one
gets to use the CPU at any given moment.

## Threads

A thread is a unit of execution that lives inside a process. A single
process can have multiple threads that run at the same time and share
the same memory, which makes it cheaper to share data between them
than between separate processes, but also means a bug in one thread
can more easily corrupt data used by another. A word processor might
use one thread to handle typing and another to run a spell check in
the background.

## Memory and Virtual Memory

Every process needs memory to store its data and instructions while it
runs. Instead of giving programs direct access to physical RAM, most
operating systems give each process the illusion of having its own
large, private block of memory, called virtual memory. The operating
system maps these virtual addresses to real physical RAM behind the
scenes, and can even temporarily move rarely used data out to disk
when RAM fills up. This is why a computer can appear to run more
programs than would actually fit in RAM at once, though performance
suffers once the disk gets involved.

## Scheduling

Since a CPU can usually only run one thing at a time per core, the
operating system has to decide, many times per second, which process
or thread runs next. This decision-making is called scheduling. A
scheduler tries to balance fairness (so no process is starved of CPU
time) with responsiveness (so the system still feels smooth to a
user). Switching the CPU from one process to another is called a
context switch, and doing it too often can itself waste time.

## File Systems

A file system is how an operating system organizes and keeps track of
files on a storage device, including their names, folder structure,
and where their data physically lives on the disk. Common examples are
NTFS on Windows and ext4 on Linux. Beyond just storing bytes, a file
system also tracks metadata such as file size, permissions, and when a
file was last modified, and it has to keep this information consistent
even if the power goes out mid-write.
