# Deploying Tools Without Server Code via SSE
Set up the conda shell hook:
```
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
```
Reload bash configuration:
```
source ~/.bashrc
```

## FMP Server
```
export FMP_API_KEY=k7akB7EWZBqNtgggXDCUgpDjBjkjb1sq
python -m src.server --sse --port 8000
```

# Deploying Tools With Server Code

## Step 1: Integrate Tools
### Local Client Import
Create a new directory under the `server` directory, each containing a tool instance decorated with fastmcp.
For example, `server/PubChem-MCP-Server/pubchem_server.py` defines an MCP tool:
```
@mcp.tool()
async def search_pubchem_by_name(name: str, max_results: int = 5) -> List[Dict[str, Any]]:
    logging.info(f"Searching for compounds with name: {name}, max_results: {max_results}")
    """
    Search for chemical compounds on PubChem using a compound name.

    Args:
        name: Name of the chemical compound
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of dictionaries containing compound information
    """
    try:
        results = await asyncio.to_thread(search_by_name, name, max_results)
        return results
    except Exception as e:
        return [{"error": f"An error occurred while searching: {str(e)}"}]
```

After defining the file, you need to specify the corresponding Python file path in `config/server_list.json` (in this case, `server/PubChem-MCP-Server/pubchem_server.py`). Additionally, some simple MCP services can be defined directly in `mcp_server.py`, which is imported by default.


### SSE Client Import
Simply add the required SSE links to `config/server_list.json`:
```
[
    "server/Agents-Server/agents_server.py",
    "server/BASE-TOOL-Server/base_tool_server.py",
    "https://dpa-uuid1750659890.app-space.dplink.cc/sse?token=b42b991d062341fba15a9f7975e190b0"
]
```

## Step 2: Enter Docker
```
docker exec -it backend_server_final /bin/bash
tmux attach -t server
```
1. Find the corresponding running server window and stop the current service:
```
pkill -f tool_server_session.py
```

2. Restart:
```
cd /mnt/tool_backends/MCP
bash deploy_server.sh
```


## Step 3: Test Deployment
Exit Docker and run the corresponding tool test file from outside the container:
```
python /home/ubuntu/shuotang/git_repo/Agent/tool_backends/test.py
```
