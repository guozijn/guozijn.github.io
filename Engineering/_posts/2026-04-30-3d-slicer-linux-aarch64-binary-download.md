---
title: "3D Slicer Linux aarch64 Binary Download"
tags:
  - 3d slicer
  - linux
  - aarch64
  - arm64
---

If you are looking for a **3D Slicer binary for Linux aarch64 / ARM64**, I have published a build here:

[Download Slicer 5.11.0 Linux aarch64 from GitHub Releases](https://github.com/guozijn/Slicer/releases/tag/v5.11.0-2026-04-28-linux-aarch64)

Direct download:

[Slicer-5.11.0-2026-04-28-linux-aarch64.tar.gz](https://github.com/guozijn/Slicer/releases/download/v5.11.0-2026-04-28-linux-aarch64/Slicer-5.11.0-2026-04-28-linux-aarch64.tar.gz)

This build is intended for people who cannot find an official or convenient **Slicer Linux aarch64 binary file** and need a ready-to-run package for an ARM64 Linux machine.

## Release Notes

This is a 3D Slicer binary file compiled for aarch64 on Linux.

Compile environment:

- Operating system: Ubuntu 24.04.4 LTS
- Architecture: aarch64
- Python version: 3.12.3

Checksum:

```text
SHA256: 944f3031f1c4840f08eb72edcdc17fb56b6f9ee82dd468a88636a46b64d17359
```

Packaging notes:

- Includes the required runtime library `libqttesting.so`.
- Removes unversioned bundled OpenSSL symlinks (`libssl.so`, `libcrypto.so`) so Qt can load Ubuntu 24.04 OpenSSL 3, while keeping versioned OpenSSL 1.1 libraries required by Slicer-linked components.

## Install

Download the archive, extract it, and run Slicer from the extracted directory:

```bash
tar -xzf Slicer-5.11.0-2026-04-28-linux-aarch64.tar.gz
cd Slicer-5.11.0-2026-04-28-linux-aarch64
./Slicer
```
