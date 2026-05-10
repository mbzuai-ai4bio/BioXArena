## FastAPI request call tool

import requests,json
import time
import os
from typing import Dict, Any
import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, '../configs/mcp_config.json'), 'r') as f:
    config = json.load(f)

url = config['mcp_server_url']

def post_item_info(session_id:str, item:Dict[str, Any]):
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "session_id":session_id,
        "item":item
    }
    resp = requests.post(
        f"{url}/put_item",
        headers=headers,
        json=payload
    )
    # print(resp.json())
    # print(resp.content)
    response = resp.json()

    return response




def call_tool(tool_name: str, tool_args: dict, session_id:str=None):
    print(f"################# session id: {session_id} ########################")
    item = {
        "main_stream_type":"tool_result",
        "sub_stream_type":"",
        "content": "",
        "from_sandbox": True,
        "stream_state": 'running',
        "other_info": {"call_tool":tool_name, "call_args":tool_args}
    }
    post_item_info(session_id, item)

    if tool_name is None:
        # List all tools
        try:
            t1 = time.time()
            resp = requests.get(f"{url}/get_tool")
            result = resp.json()
            t2 = time.time()
            return {
                "tool_result": result,
                "tool_elapsed_time": t2 - t1
            }
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    else:
        try:
            t1 = time.time()
            resp = requests.post(
                f"{url}/call_tool/{tool_name}",
                json=tool_args
            )
            result = resp.json()
            if result["status"]:
                t2 = time.time()
                return {
                    "tool_result": result["result"],
                    "tool_elapsed_time": t2 - t1
                }
            else:
                print(f"Tool error: {result['result']}")
                return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None

def code_tool(code:str, timeout=1800):
    try:
        resp = requests.post(
            f"{url}/execute",
            json={"code":code, "timeout": timeout},
        )
        result = resp.json()
        return result
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def test():
    code = """
print(browse_master("1+1=?,do not use web_parse"))
"""
    
    result = code_tool(code)
    print(result)

import time
if __name__ == "__main__":
    test()
    exit()
    result = call_tool("web_parse_nano", {'link': 'https://bohr.physics.berkeley.edu/classes/221/1112/notes/covariance.pdf', 'query': "What is main content of this page?"})
    print(result)

    
    