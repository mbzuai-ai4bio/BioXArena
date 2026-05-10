# Calculation Path Adaptor (bohr-agent-sdk)

Consistent with _tmp/MatMaster: **HTTPS storage uses Bohrium authentication**; **executor** is distinguished by "sync/async": synchronous tasks pass `None`, while the rest pass a Bohrium executor with the specified image/machine type (authentication is injected from .env).

## Injected Parameters

- **executor** (per `mcp.calculation_executors` config):
  - If the tool name is in the server's `sync_tools` list -> passes `None` (synchronous execution, running in the server's default environment).
  - Otherwise, if the server has an `executor` template configured (image/machine type) -> passes a Bohrium executor (authentication injected by `evomaster.env.inject_bohrium_executor` from .env).
  - No config or no template -> `None`.
- **storage**: `get_bohrium_storage_config()` (from `evomaster.env.bohrium`), reads `BOHRIUM_ACCESS_KEY` and `BOHRIUM_PROJECT_ID` from `.env`.
- **Input paths**: Local/workspace files are uploaded to OSS (when configured) and replaced with HTTPS URLs before calling MCP.

## /workspace Mapping

The Agent may pass paths like `/workspace/Fe_bcc.cif`. This adaptor maps `/workspace/` to the current session's `workspace_path`, i.e., `workspace_path/Fe_bcc.cif`, then checks whether the file exists and uploads it to OSS.

## Dependencies

- **Runtime environment**: Same as the process running `python run.py`. ConfigManager loads `.env` from the **project root directory** when loading configuration, so OSS/Bohrium-related variables must be set in the **project root's .env** file (or exported in the shell of that process).
- Environment variables: `OSS_ENDPOINT`, `OSS_BUCKET_NAME`, `OSS_ACCESS_KEY_ID`, `OSS_ACCESS_KEY_SECRET` (required when uploading local files to OSS); Bohrium authentication as described above.
- `pip install oss2` (already included in the main dependencies).
