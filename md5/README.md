# MD5
MD5 hash algorithm implementation.

[From MD5 RFC 1321]
The  algorithm takes as input a message of arbitrary length and produces as output a 128-bit "fingerprint" or "message digest" of the input. It is conjectured that it is computationally infeasible to produce two messages having the same message digest, or to produce any message having a given prespecified target message digest. The MD5 algorithm is intended for digital signature applications, where a large file must be "compressed" in a secure manner before being encrypted with a private (secret) key under a public-key cryptosystem such as RSA.

## Getting Started

To simply compile and run without installing just run:

```bash
$ gcc src/md5.c
$ ./md5 --help
```

The following instructions is for compiling and installing this project.

### Prerequisites

- GCC
- Autotools

E.g. On Debian-based distributions:

```bash
$ sudo apt install gcc autotools
```

### Installation

1. Clone repo
2. Set up an m4 environment
3. Generate configure file fron configure.ac
4. Generate Makefile.in from Makefile.am
5. Generate Makefile from Makefile.in
6. Build project
7. Install

```bash
$ git clone https://github.com/pskenny/md5 # 1
$ cd md5
$ aclocal # 2
$ autoconf # 3
$ automake --add-missing # 4
$ ./configure # 5
$ make # 6
$ make install # 7
```

## Resources

- Theory Of Algorithms module course work (including documents, videos and lectures) provided by [Ian McLoughlin](https://github.com/ianmcloughlin/).
- [The magic behind configure, make, make install](https://thoughtbot.com/blog/the-magic-behind-configure-make-make-install). How to use autotools for easier project build, distribution and installation.
