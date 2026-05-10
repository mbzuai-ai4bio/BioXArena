# Tool Import Guide

## Local Client Import
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


## SSE Client Import
Simply add the required SSE links to `config/server_list.json`:
```
[
    "server/Agents-Server/agents_server.py",
    "server/BASE-TOOL-Server/base_tool_server.py",
    "https://dpa-uuid1750659890.app-space.dplink.cc/sse?token=b42b991d062341fba15a9f7975e190b0"
]
```

## Running the Service
It is recommended to run the service inside a Docker container. We have our own Docker image. Reference startup script:
```
docker run -d \
  --name backend_server_shenshi \
  --cpuset-cpus="64-95" \
  -p 30004:30004 \
  -v #path-to-agent-directory:/mnt \
  fastapi-server \
  tail -f /dev/null
```
CPU pinning is used here to improve multi-core performance and increase concurrency.
Run the deployment script inside the container:
```
bash deploy_server.sh
```

## Calling the Service

### Using the Agent Framework's Built-in Tool Manager to Execute Code (Recommended)

```
class StreamToolManager(BaseToolManager):
    def __init__(self, url, session_id:str = None, timeout:int=180):
        super().__init__(url)
        self.session_id = str(uuid4()) if not session_id else session_id
        self.timeout = timeout

    async def submit_task(self, code:str):
        ...

    async def recieve_task_process(self, ):
        ...

    async def execute_code_async_stream(self, tool_call: str,):
        submit_status = await self.submit_task(tool_call)
        if submit_status["status"] == "fail":
            yield {"output":""}
            return

        async for item in self.recieve_task_process():
            yield item

    async def close_session(self):
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.server_url}/del_session",
                params={"session_id": self.session_id}
            )
            return resp.json()
```
- You need to maintain your own tool manager for each session ID. Each session ID corresponds to an independent code execution space where historical functions and variables are preserved.
- After the entire session inference is complete, if you no longer need the session, you must manually delete it (using the `close_session()` function) to free the code execution memory and variables for that session.
- Use `execute_code_async_stream` to execute code. The execution results appear in a streaming fashion, with the following scenarios:

1. When the agent calls a tool directly and the tool returns normally, all called tools and the final code execution result are streamed back. When a tool returns, `other_info` contains a dictionary of `function_name: result`. When code returns, `main_stream_type` is `code_result`. `stream_state` has three values: `start`, `running`, and `end`.
```
    {
        "main_stream_type": "tool_result",
        "sub_stream_type": "",
        "content": "",
        "from_sandbox": true,
        "stream_state": "start",
        "other_info": {}
    },
    {
        "main_stream_type": "tool_result",
        "sub_stream_type": "",
        "content": "",
        "from_sandbox": true,
        "stream_state": "running",
        "other_info": {
            "web_search": {
                "tool_result": {
                    "organic": [
                        {
                            "title": "Buy iPhone 15 and iPhone 15 Plus - Apple",
                            "link": "https://www.apple.com/shop/buy-iphone/iphone-15",
                            "snippet": "BuyFrom $799.00Footnote**Pay with Apple Pay or other payment methods. Finance ...",
                            "sitelinks": [
                                {
                                    "title": "Buy iPhone 15",
                                    "link": "https://www.apple.com/shop/buy-iphone/iphone-15/6.1-inch-display-128gb-black-unlocked"
                                }
                            ],
                            "position": 1
                        },
                        {
                            "title": "Buy iPhone - Apple",
                            "link": "https://www.apple.com/shop/buy-iphone",
                            "snippet": "Shop the latest iPhone models and accessories. Save with Apple Trade In, carrier offers, and flexible monthly payment options. Get expert help today.",
                            "position": 2
                        },
                        {
                            "title": "iPhone - Apple",
                            "link": "https://www.apple.com/iphone/",
                            "snippet": "The ultimate iPhone ... From $999 or $41.62/mo. for 24 mo ... Learn more Buy ... Apple Intelligence.",
                            "position": 3
                        },
                        {
                            "title": "Apple Store - Find a Store",
                            "link": "https://www.apple.com/retail/",
                            "snippet": "Activation required. AT&T iPhone 15 Special Deal: Buy an iPhone 15 128 GB and get $441.36 in bill credits applied over 36 months. Buy an iPhone 15 256 GB and ...",
                            "position": 4
                        },
                        {
                            "title": "Apple iPhone 15: Order, Price, Colors, Features - Verizon",
                            "link": "https://www.verizon.com/smartphones/apple-iphone-15/",
                            "snippet": "You'll pay $0.00/mo after a credit of $20.27/mo on your billed price of $20.27/mo per device for 36 mos. Full retail price $729.99. Your total payments will ...",
                            "position": 5
                        },
                        {
                            "title": "iPhone - Buying iPhone - Apple",
                            "link": "https://www.apple.com/iphone/buy/",
                            "snippet": "Purchase your next iPhone from the Apple Store. Get answers about carriers ... Greatest price. From $599. Buy \u00b7 Learn more \u00b7 View in AR. 6.1\u2033. Super Retina ...",
                            "position": 6
                        },
                        {
                            "title": "Amazon.com: Apple iPhone 15 Pro Max, 256GB, Natural Titanium",
                            "link": "https://www.amazon.com/Apple-iPhone-15-Pro-Max/dp/B0CMZD7VCV",
                            "snippet": "List Price: $829.00$829.00 Details. The List Price is the suggested retail price of a new product as provided by a manufacturer, supplier, or seller. Except ...",
                            "rating": 3.9,
                            "ratingCount": 1511,
                            "position": 7
                        },
                        {
                            "title": "Interesting to see the USD price difference of the iPhone 15 Pro ...",
                            "link": "https://www.reddit.com/r/iphone/comments/16kl59k/interesting_to_see_the_usd_price_difference_of/",
                            "snippet": "1K votes, 332 comments. For fun I priced out a top-spec 15 Pro Max in Turkey and it comes out to US$3458.88. Cheap!",
                            "date": "Sep 17, 2023",
                            "position": 8
                        },
                        {
                            "title": "Apple",
                            "link": "https://www.apple.com/",
                            "snippet": "Discover the innovative world of Apple and shop everything iPhone, iPad, Apple Watch, Mac, and Apple TV, plus explore accessories, entertainment, ...",
                            "position": 9
                        }
                    ],
                    "peopleAlsoAsk": [
                        {
                            "question": "What is the iPhone 15 going to cost?",
                            "snippet": "New or existing customer Retail price: $729.99. One-time activation fee of $35.",
                            "title": "Apple iPhone 15: Order, Price, Colors, Features - Verizon",
                            "link": "https://www.verizon.com/smartphones/apple-iphone-15/"
                        },
                        {
                            "question": "How much will Apple pay for an iPhone 15?",
                            "snippet": "Get $45\u2013$600 for your trade-in. Get 3% Daily Cash back with Apple Card.",
                            "title": "Buy iPhone 15 and iPhone 15 Plus - Apple",
                            "link": "https://www.apple.com/shop/buy-iphone/iphone-15"
                        },
                        {
                            "question": "How much will I get for an iPhone 15?",
                            "snippet": "Get \u00a330-\u00a3595 when you trade in an iPhone. 1\nYour device\nEstimated trade-in value 1\niPhone 15 Pro Max\nUp to \u00a3595\niPhone 15 Pro\nUp to \u00a3535\niPhone 15 Plus\nUp to \u00a3420\niPhone 15\nUp to \u00a3385",
                            "title": "Apple Trade In - Apple (UK)",
                            "link": "https://www.apple.com/uk/shop/trade-in"
                        },
                        {
                            "question": "How much did the iPhone 15 originally cost?",
                            "snippet": "Pricing and Availability iPhone 15 and iPhone 15 Plus will be available in pink, yellow, green, blue, and black in 128GB, 256GB, and 512GB storage capacities, starting at $799 (U.S.) or $33.29 (U.S.) per month, and $899 (U.S.) or $37.45 (U.S.) per month, respectively.",
                            "title": "Apple debuts iPhone 15 and iPhone 15 Plus",
                            "link": "https://www.apple.com/newsroom/2023/09/apple-debuts-iphone-15-and-iphone-15-plus/"
                        }
                    ],
                    "relatedSearches": [
                        {
                            "query": "Apple IPhone 13"
                        },
                        {
                            "query": "Iphone 15 price official apple store usa"
                        },
                        {
                            "query": "Iphone 15 price official apple store near me"
                        },
                        {
                            "query": "iPhone 15 Pro Max"
                        },
                        {
                            "query": "iPhone 15 price in USA"
                        },
                        {
                            "query": "iPhone 15 Pro Max price"
                        },
                        {
                            "query": "iPhone 15 Pro Max price in USA"
                        },
                        {
                            "query": "iPhone 14"
                        },
                        {
                            "query": "iPhone 15 Pro price in USA"
                        }
                    ]
                },
                "tool_elapsed_time": 2.1127848625183105
            }
        }
    },
    {
        "main_stream_type": "code_result",
        "sub_stream_type": "",
        "content": "{'tool_result': {'organic': [{'title': 'Buy iPhone 15 and iPhone 15 Plus - Apple', 'link': 'https://www.apple.com/shop/buy-iphone/iphone-15', 'snippet': 'BuyFrom $799.00Footnote**Pay with Apple Pay or other payment methods. Finance ...', 'sitelinks': [{'title': 'Buy iPhone 15', 'link': 'https://www.apple.com/shop/buy-iphone/iphone-15/6.1-inch-display-128gb-black-unlocked'}], 'position': 1}, {'title': 'Buy iPhone - Apple', 'link': 'https://www.apple.com/shop/buy-iphone', 'snippet': 'Shop the latest iPhone models and accessories. Save with Apple Trade In, carrier offers, and flexible monthly payment options. Get expert help today.', 'position': 2}, {'title': 'iPhone - Apple', 'link': 'https://www.apple.com/iphone/', 'snippet': 'The ultimate iPhone ... From $999 or $41.62/mo. for 24 mo ... Learn more Buy ... Apple Intelligence.', 'position': 3}, {'title': 'Apple Store - Find a Store', 'link': 'https://www.apple.com/retail/', 'snippet': 'Activation required. AT&T iPhone 15 Special Deal: Buy an iPhone 15 128 GB and get $441.36 in bill credits applied over 36 months. Buy an iPhone 15 256 GB and ...', 'position': 4}, {'title': 'Apple iPhone 15: Order, Price, Colors, Features - Verizon', 'link': 'https://www.verizon.com/smartphones/apple-iphone-15/', 'snippet': \"You'll pay $0.00/mo after a credit of $20.27/mo on your billed price of $20.27/mo per device for 36 mos. Full retail price $729.99. Your total payments will ...\", 'position': 5}, {'title': 'iPhone - Buying iPhone - Apple', 'link': 'https://www.apple.com/iphone/buy/', 'snippet': 'Purchase your next iPhone from the Apple Store. Get answers about carriers ... Greatest price. From $599. Buy \u00b7 Learn more \u00b7 View in AR. 6.1\u2033. Super Retina ...', 'position': 6}, {'title': 'Amazon.com: Apple iPhone 15 Pro Max, 256GB, Natural Titanium', 'link': 'https://www.amazon.com/Apple-iPhone-15-Pro-Max/dp/B0CMZD7VCV', 'snippet': 'List Price: $829.00$829.00 Details. The List Price is the suggested retail price of a new product as provided by a manufacturer, supplier, or seller. Except ...', 'rating': 3.9, 'ratingCount': 1511, 'position': 7}, {'title': 'Interesting to see the USD price difference of the iPhone 15 Pro ...', 'link': 'https://www.reddit.com/r/iphone/comments/16kl59k/interesting_to_see_the_usd_price_difference_of/', 'snippet': '1K votes, 332 comments. For fun I priced out a top-spec 15 Pro Max in Turkey and it comes out to US$3458.88. Cheap!', 'date': 'Sep 17, 2023', 'position': 8}, {'title': 'Apple', 'link': 'https://www.apple.com/', 'snippet': 'Discover the innovative world of Apple and shop everything iPhone, iPad, Apple Watch, Mac, and Apple TV, plus explore accessories, entertainment, ...', 'position': 9}], 'peopleAlsoAsk': [{'question': 'What is the iPhone 15 going to cost?', 'snippet': 'New or existing customer Retail price: $729.99. One-time activation fee of $35.', 'title': 'Apple iPhone 15: Order, Price, Colors, Features - Verizon', 'link': 'https://www.verizon.com/smartphones/apple-iphone-15/'}, {'question': 'How much will Apple pay for an iPhone 15?', 'snippet': 'Get $45\u2013$600 for your trade-in. Get 3% Daily Cash back with Apple Card.', 'title': 'Buy iPhone 15 and iPhone 15 Plus - Apple', 'link': 'https://www.apple.com/shop/buy-iphone/iphone-15'}, {'question': 'How much will I get for an iPhone 15?', 'snippet': 'Get \u00a330-\u00a3595 when you trade in an iPhone. 1\\nYour device\\nEstimated trade-in value 1\\niPhone 15 Pro Max\\nUp to \u00a3595\\niPhone 15 Pro\\nUp to \u00a3535\\niPhone 15 Plus\\nUp to \u00a3420\\niPhone 15\\nUp to \u00a3385', 'title': 'Apple Trade In - Apple (UK)', 'link': 'https://www.apple.com/uk/shop/trade-in'}, {'question': 'How much did the iPhone 15 originally cost?', 'snippet': 'Pricing and Availability iPhone 15 and iPhone 15 Plus will be available in pink, yellow, green, blue, and black in 128GB, 256GB, and 512GB storage capacities, starting at $799 (U.S.) or $33.29 (U.S.) per month, and $899 (U.S.) or $37.45 (U.S.) per month, respectively.', 'title': 'Apple debuts iPhone 15 and iPhone 15 Plus', 'link': 'https://www.apple.com/newsroom/2023/09/apple-debuts-iphone-15-and-iphone-15-plus/'}], 'relatedSearches': [{'query': 'Apple IPhone 13'}, {'query': 'Iphone 15 price official apple store usa'}, {'query': 'Iphone 15 price official apple store near me'}, {'query': 'iPhone 15 Pro Max'}, {'query': 'iPhone 15 price in USA'}, {'query': 'iPhone 15 Pro Max price'}, {'query': 'iPhone 15 Pro Max price in USA'}, {'query': 'iPhone 14'}, {'query': 'iPhone 15 Pro price in USA'}]}, 'tool_elapsed_time': 2.1127848625183105}\n",
        "from_sandbox": true,
        "stream_state": "running",
        "other_info": {}
    },
    {
        "main_stream_type": "tool_result",
        "sub_stream_type": "",
        "content": "",
        "from_sandbox": true,
        "stream_state": "end",
        "other_info": {}
    },
```
2. When an agent calls another agent as a tool, the other agent's text and tool call results are streamed back.
```
{
    "main_stream_type": "tool_result",
    "sub_stream_type": "text",
    "content": ">\n\n",
    "from_sandbox": true,
    "stream_state": "running",
    "other_info": {}
},
{
    "main_stream_type": "tool_result",
    "sub_stream_type": "tool_result",
    "content": "",
    "from_sandbox": true,
    "stream_state": "start",
    "other_info": {}
},
{
    "main_stream_type": "tool_result",
    "sub_stream_type": "tool_result",
    "content": "",
    "from_sandbox": true,
    "stream_state": "running",
    "other_info": {
        "web_search": {
            "tool_result": {
                "organic": [
                    {
                        "title": "Buy iPhone 16 and iPhone 16 Plus - Apple",
                        "link": "https://www.apple.com/shop/buy-iphone/iphone-16",
                        "snippet": "iPhone 16, Ultramarine finish, back exterior, top rounded corners, advanced dual-camera system, flash, microphone. From $799 or $33.29/mo. per month for 24 mo. ...",
                        "position": 1
                    },
                    {
                        "title": "iPhone 16 and iPhone 16 Plus - Apple",
                        "link": "https://www.apple.com/iphone-16/",
                        "snippet": "iPhone 16 and iPhone 16 Plus. Built for Apple Intelligence. Camera Control. 48MP Fusion camera. Five vibrant colors. A18 chip.",
                        "position": 2
                    },
                    {
                        "title": "Buy iPhone 16 Plus 128GB Black - Apple",
                        "link": "https://www.apple.com/shop/buy-iphone/iphone-16/6.7-inch-display-128gb-black-unlocked",
                        "snippet": "iPhone 16 Plus 128GB Black. $929.00. One-time payment. Get 3% Daily Cash with Apple Card. Add to Bag. Order now. Pick up, in store: Today at Apple Knox Street.",
                        "position": 3
                    },
                    {
                        "title": "Buy iPhone 16 Plus 128GB White - Apple",
                        "link": "https://www.apple.com/shop/buy-iphone/iphone-16/6.7-inch-display-128gb-white-unlocked",
                        "snippet": "Get $45 - $600 off a new iPhone 16 or iPhone 16 Plus when you trade in an iPhone 8 or newer. 0% financing available. Buy now with free shipping.",
                        "sitelinks": [
                            {
                                "title": "Model. Which Is Best For You...",
                                "link": "https://www.apple.com/shop/buy-iphone/iphone-16/6.7-inch-display-128gb-white-unlocked#:~:text=Model.%20Which%20is%20best%20for%20you%3F"
                            },
                            {
                                "title": "Connectivity. Choose A...",
                                "link": "https://www.apple.com/shop/buy-iphone/iphone-16/6.7-inch-display-128gb-white-unlocked#:~:text=Connectivity.%20Choose%20a%20carrier."
                            },
                            {
                                "title": "What's In The Box",
                                "link": "https://www.apple.com/shop/buy-iphone/iphone-16/6.7-inch-display-128gb-white-unlocked#:~:text=What%27s%20in%20the%20Box"
                            }
                        ],
                        "position": 4
                    },
                    {
                        "title": "Buy iPhone 16 Plus 256GB Black - Apple",
                        "link": "https://www.apple.com/shop/buy-iphone/iphone-16/6.7-inch-display-256gb-black-unlocked",
                        "snippet": "Get $45 - $600 off a new iPhone 16 or iPhone 16 Plus when you trade in an iPhone 8 or newer. 0% financing available. Buy now with free shipping.",
                        "sitelinks": [
                            {
                                "title": "Model. Which Is Best For You...",
                                "link": "https://www.apple.com/shop/buy-iphone/iphone-16/6.7-inch-display-256gb-black-unlocked#:~:text=Model.%20Which%20is%20best%20for%20you%3F"
                            },
                            {
                                "title": "Connectivity. Choose A...",
                                "link": "https://www.apple.com/shop/buy-iphone/iphone-16/6.7-inch-display-256gb-black-unlocked#:~:text=Connectivity.%20Choose%20a%20carrier."
                            },
                            {
                                "title": "What's In The Box",
                                "link": "https://www.apple.com/shop/buy-iphone/iphone-16/6.7-inch-display-256gb-black-unlocked#:~:text=What%27s%20in%20the%20Box"
                            }
                        ],
                        "position": 5
                    },
                    {
                        "title": "Buy iPhone - Apple",
                        "link": "https://www.apple.com/shop/buy-iphone",
                        "snippet": "Shop the latest iPhone models and accessories. Save with Apple Trade In, carrier offers, and flexible monthly payment options. Get expert help today.",
                        "position": 6
                    },
                    {
                        "title": "iPhone - Apple",
                        "link": "https://www.apple.com/iphone/",
                        "snippet": "Designed for Apple Intelligence. Discover the iPhone 16e along with iPhone 16 Pro, iPhone 16, and iPhone 15 ... Greatest price. From $599 or $24.95/mo ...",
                        "position": 7
                    },
                    {
                        "title": "Buy iPhone 16 and iPhone 16 Plus - Education - Apple",
                        "link": "https://www.apple.com/us-edu/shop/buy-iphone/iphone-16",
                        "snippet": "iPhone 16, Ultramarine finish, back exterior, top rounded corners, advanced dual-camera system, flash, microphone. From $799 or $33.29/mo. per month for 24 mo. ...",
                        "sitelinks": [
                            {
                                "title": "Model. Which Is Best For You...",
                                "link": "https://www.apple.com/us-edu/shop/buy-iphone/iphone-16#:~:text=Model.%20Which%20is%20best%20for%20you%3F"
                            },
                            {
                                "title": "Connectivity. Choose A...",
                                "link": "https://www.apple.com/us-edu/shop/buy-iphone/iphone-16#:~:text=Connectivity.%20Choose%20a%20carrier."
                            },
                            {
                                "title": "Applecare+ Coverage. Protect...",
                                "link": "https://www.apple.com/us-edu/shop/buy-iphone/iphone-16#:~:text=AppleCare%2B%20coverage.%20Protect%20your%20new%20iPhone."
                            }
                        ],
                        "position": 8
                    },
                    {
                        "title": "Apple introduces iPhone 16 and iPhone 16 Plus",
                        "link": "https://www.apple.com/newsroom/2024/09/apple-introduces-iphone-16-and-iphone-16-plus/",
                        "snippet": "iPhone 16 and iPhone 16 Plus are built for Apple Intelligence, and feature Camera Control, the Action button, a 48MP Fusion camera, and the A18 chip.",
                        "date": "Sep 9, 2024",
                        "position": 9
                    },
                    {
                        "title": "Buy iPhone 16 Plus 256GB Pink - Apple",
                        "link": "https://www.apple.com/shop/buy-iphone/iphone-16/6.7-inch-display-256gb-pink-unlocked",
                        "snippet": "Latest iPhone. Greatest price. From $599. 6.1\u2033. Super Retina XDR display footnote \u00b9. \u2014 No ProMotion technology. \u2014 ...",
                        "sitelinks": [
                            {
                                "title": "Model. Which Is Best For You...",
                                "link": "https://www.apple.com/shop/buy-iphone/iphone-16/6.7-inch-display-256gb-pink-unlocked#:~:text=Model.%20Which%20is%20best%20for%20you%3F"
                            },
                            {
                                "title": "Connectivity. Choose A...",
                                "link": "https://www.apple.com/shop/buy-iphone/iphone-16/6.7-inch-display-256gb-pink-unlocked#:~:text=Connectivity.%20Choose%20a%20carrier."
                            },
                            {
                                "title": "What's In The Box",
                                "link": "https://www.apple.com/shop/buy-iphone/iphone-16/6.7-inch-display-256gb-pink-unlocked#:~:text=What%27s%20in%20the%20Box"
                            }
                        ],
                        "position": 10
                    }
                ],
                "peopleAlsoAsk": [
                    {
                        "question": "How much is the iPhone 16 Plus at the Apple Store?",
                        "snippet": "BuyFrom $899.00**Pay with Apple Pay or other payment methods. Finance$37.45/mo.",
                        "title": "Buy iPhone 16 Plus 128GB Black - Apple",
                        "link": "https://www.apple.com/shop/buy-iphone/iphone-16/6.7-inch-display-128gb-black-unlocked"
                    },
                    {
                        "question": "How much will the iPhone 16 Plus be?",
                        "snippet": "Tariffs for iPhone 16 Plus:\nMonthly phone payment\n\u00a322.24\nCash price\n\u00a3800.64\nCredit amount\n\u00a3800.64\nInterest rate (fixed)\n0%\nAPR representative\n0%",
                        "title": "iPhone 16 Plus | Pay Monthly Deals & Contracts - Tesco Mobile",
                        "link": "https://www.tescomobile.com/shop/apple/iphone-16-plus"
                    },
                    {
                        "question": "How to get iPhone 16 cheapest?",
                        "snippet": "Signing up for Boost Mobile's $60 per month Unlimited Premium plan saves you $500 on the iPhone 16E, and the Infinite Access plan for $65 a month saves up to $1,000 on the other models, which will net you the basic iPhone 16, 16 Plus and 16 Pro for free and the 16 Pro Max for less than $6 per month.",
                        "title": "Best iPhone 16 Deals: Grab a New Phone for Free Thanks to ... - CNET",
                        "link": "https://www.cnet.com/deals/best-iphone-16-deals/"
                    },
                    {
                        "question": "How much is an iPhone 16 Plus 256GB in the USA?",
                        "snippet": "BuyFrom $999.00**Pay with Apple Pay or other payment methods.",
                        "title": "Buy iPhone 16 Plus 256GB Black - Apple",
                        "link": "https://www.apple.com/shop/buy-iphone/iphone-16/6.7-inch-display-256gb-black-unlocked"
                    }
                ],
                "relatedSearches": [
                    {
                        "query": "Apple iphone 16 plus price official site unlocked"
                    },
                    {
                        "query": "Apple iphone 16 plus price official site usa"
                    }
                ]
            },
            "tool_elapsed_time": 1.5962419509887695
        }
    }
},
```
### Direct HTTP Request to the Execute Endpoint

Send a code execution request to the target service via HTTP. Assuming the tool service is deployed on port 30004:
```
import requests
import time
url = "http://127.0.0.1:30004"

test_code = f"""
from tools import *
link = "https://www2.census.gov/library/publications/2001/demo/p60-214.html"
query = "Does this document mention any towns with 0% poverty rate for 65+ population and the specified income figures?"
result = web_parse_qwen(link, query)
print(result)
"""

headers = {
    "Content-Type": "application/json"
}


def code_tool(code:str):
    start_time = time.time()
    payload = {
        "code":code
    }
    resp = requests.post(
        f"{url}/execute",
        headers=headers,
        json=payload
    )
    print(resp.content)
    response = resp.json()
    elapsed = time.time() - start_time
    response['total_time'] = elapsed
    response['server_time'] = resp.elapsed.total_seconds()

    return response

code_tool(test_code)

print([test_code])

```
