---
layout: post
tag-name: mpi
title: Message Passing Interface
---

## What is MPI
MPI is a standard library for message passing.
## Why MPI
 - **The MPI interface is designed and implemented with performance in mind.**
 - **The MPI interface will take advantage of the fastest network transport available to it when sending messages.**<br>For an instance, to communicate with two different processes within a node, MPI will use a shared memory(within a computer) to send and receive messages rather than network communications. On fast interconnects, within a high performance computer cluster it already knows how to take advantage of transports like Infiniband or Myrinet for communications to processes on other nodes, and if all else fails only it will use the Standard Internet TCP/IP.
 - **MPI Enforces other guarantees.**
 Retries, messages arrive in order.
 - **MPI is designed for multi-node technical computing.**
We can spend our time figuring out how to decompose our scientific problem rather than having to worry about network protocols.

## Several ways of communication
### Broadcast (one to many)
One process has a piece of data and broadcasts it to many or all of the other processes. All other processes will receive the same piece of data.

<div align=center><img width="700" height="300" src="https://raw.githubusercontent.com/SharynHu/picBed/master/85B15782-2C57-4395-9385-A57FD9CDF2AB.png"/></div>

### Scatter (one to many)
A close relative of the broadcast is Scatter, where one process divides values between many others. Different processes may receive different pieces of data.

<div align=center><img width="700" height="300" src="https://raw.githubusercontent.com/SharynHu/picBed/master/DD3FFC74-B93C-4CE1-B4D9-9DB8E6EA8603.png"/></div>

### Gather (many to one)
The inverse of Scatter is Gather. In which many processes have different parts of the overall picture which are then brought together to one process.

<div align=center><img width="700" height="300" src="https://raw.githubusercontent.com/SharynHu/picBed/master/C6EE0C1B-A13A-4C44-B1D2-965737EF3E11.png"/></div>

### Reduction (one to many)
Reduction which combines communication and computation of many useful sorts of operations finding a global minimum, maximum sum or product are fundamentally reduction operations. Consider doing a global sum of data. Each process calculates its partial sum and then these are combined into a global sum on one process.

<div align=center><img width="700" height="300" src="https://raw.githubusercontent.com/SharynHu/picBed/master/C6EE0C1B-A13A-4C44-B1D2-965737EF3E11.png"/></div>

<div align=center><img width="700" height="300" src="https://raw.githubusercontent.com/SharynHu/picBed/master/E516E150-03BC-4AFB-A148-589AD21BA087.png"/></div>
