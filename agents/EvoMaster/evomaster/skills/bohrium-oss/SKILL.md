---
name: bohrium-oss
description: Guide for uploading local files to Bohrium OSS when MCP tools require file URLs. Use this when you need to transmit local files through MCP tools that cannot accept local paths directly.
license: Proprietary
---

# Bohrium OSS File Upload Guide

## Overview

When calling MCP tools that require file transmission, you cannot use local file paths directly. You must upload files to Bohrium OSS first and use the returned URL in the MCP tool. This guide covers the setup and upload workflow.

## Prerequisites

### Environment Variables

Ensure the following environment variables are set before uploading:

| Variable | Description |
|----------|-------------|
| `HTTP_PLUGIN_TYPE` | HTTP plugin configuration |
| `BOHRIUM_USER_ID` | Bohrium user ID |
| `BOHRIUM_EMAIL` | Bohrium account email |
| `BOHRIUM_PASSWORD` | Bohrium account password |
| `BOHRIUM_PROJECT_ID` | Bohrium project ID |
| `BOHRIUM_ACCESS_KEY` | Bohrium access key |

If any of these are not set, remind the user to configure them.

### Install bohr-agent (Optional)

bohr-agent is typically pre-installed. If not available:

```bash
pip install bohr-agent-sdk -i https://pypi.org/simple --upgrade
```

## Quick Start

```bash
bohr-agent artifact upload -s https path/to/file
```

After the upload completes, you will receive a URL. Use this URL in the MCP tool's file parameter—users and MCP servers can access the file directly via this URL.

## Workflow

1. **Verify environment** — Check that all required environment variables are set
2. **Upload file** — Run `bohr-agent artifact upload -s https <file_path>`
3. **Use URL** — Copy the returned URL and paste it into the MCP tool's file/URL field

## Quick Reference

| Task | Command |
|------|---------|
| Upload single file | `bohr-agent artifact upload -s https path/to/file` |
| Install bohr-agent | `pip install bohr-agent-sdk -i https://pypi.org/simple --upgrade` |
