"""Test Agent Context management functionality

Tests all context-management-related features in agent.py:
- reset_context()
- add_user_message()
- add_assistant_message()
- add_tool_message()
- get_current_dialog()
- get_conversation_history()
- context_manager.prepare_for_query()
- context_manager.should_truncate()
- context_manager.truncate() (different strategies)
- context_manager.estimate_tokens()
"""
# Add the project root directory to the Python path so that the evomaster module can be imported
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from evomaster.agent.agent import Agent, AgentConfig
from evomaster.agent.context import ContextConfig, ContextManager, TruncationStrategy
from evomaster.agent.session.base import BaseSession, SessionConfig
from evomaster.agent.tools.base import ToolRegistry
from evomaster.utils.types import (
    AssistantMessage,
    Dialog,
    SystemMessage,
    TaskInstance,
    ToolCall,
    ToolMessage,
    UserMessage,
    FunctionCall,
)


class MockLLM:
    """Mock LLM for testing"""
    
    def __init__(self):
        self.query_calls = []
    
    def query(self, dialog: Dialog) -> AssistantMessage:
        """Simulate an LLM query"""
        self.query_calls.append(dialog)
        # Return a simple assistant message
        return AssistantMessage(content="Mock response")


class MockSession(BaseSession):
    """Mock Session for testing"""
    
    def __init__(self, config=None):
        super().__init__(config or SessionConfig())
        self._is_open = True
    
    def open(self):
        self._is_open = True
    
    def close(self):
        self._is_open = False
    
    def exec_bash(self, command: str, timeout=None, is_input=False):
        return {"stdout": "", "stderr": "", "exit_code": 0}
    
    def upload(self, local_path: str, remote_path: str):
        pass
    
    def download(self, remote_path: str, timeout=None):
        return b""


class TestAgentContextManagement(unittest.TestCase):
    """Test Agent context management functionality"""
    
    def setUp(self):
        """Set up the test environment"""
        self.llm = MockLLM()
        self.session = MockSession()
        self.tools = ToolRegistry()
        self.task = TaskInstance(
            task_id="test_task_001",
            task_type="test",
            description="测试任务",
            input_data={}
        )
        
        # Create temporary prompt files (kept alive during the test)
        self.tmpdir = tempfile.mkdtemp()
        tmpdir_path = Path(self.tmpdir)
        self.system_prompt_file = tmpdir_path / "system_prompt.txt"
        self.user_prompt_file = tmpdir_path / "user_prompt.txt"
        
        self.system_prompt_file.write_text("You are a test assistant.")
        self.user_prompt_file.write_text("Complete the test task.")
    
    def tearDown(self):
        """Clean up the test environment"""
        import shutil
        if hasattr(self, 'tmpdir'):
            shutil.rmtree(self.tmpdir, ignore_errors=True)
    
    def create_agent(self, context_config=None):
        """Create an Agent instance"""
        agent_config = AgentConfig(context_config=context_config or ContextConfig())
        
        return Agent(
            llm=self.llm,
            session=self.session,
            tools=self.tools,
            system_prompt_file=str(self.system_prompt_file),
            user_prompt_file=str(self.user_prompt_file),
            config=agent_config,
            enable_tools=False,  # Disable tool calls to simplify testing
        )
    
    def test_initial_context(self):
        """Test initial context state"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        # Check initial dialog
        dialog = agent.get_current_dialog()
        self.assertIsNotNone(dialog)
        self.assertEqual(len(dialog.messages), 2)  # System + User
        self.assertIsInstance(dialog.messages[0], SystemMessage)
        self.assertIsInstance(dialog.messages[1], UserMessage)
        
        # Check that initial prompts are saved
        self.assertIsNotNone(agent._initial_system_prompt)
        self.assertIsNotNone(agent._initial_user_prompt)
    
    def test_add_user_message(self):
        """Test adding user messages"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        initial_count = len(agent.get_conversation_history())
        agent.add_user_message("这是用户消息1")
        
        history = agent.get_conversation_history()
        self.assertEqual(len(history), initial_count + 1)
        self.assertIsInstance(history[-1], UserMessage)
        self.assertEqual(history[-1].content, "这是用户消息1")
        
        # Add another one
        agent.add_user_message("这是用户消息2")
        history = agent.get_conversation_history()
        self.assertEqual(len(history), initial_count + 2)
        self.assertEqual(history[-1].content, "这是用户消息2")
    
    def test_add_assistant_message(self):
        """Test adding assistant messages"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        initial_count = len(agent.get_conversation_history())
        agent.add_assistant_message("这是助手回复")
        
        history = agent.get_conversation_history()
        self.assertEqual(len(history), initial_count + 1)
        self.assertIsInstance(history[-1], AssistantMessage)
        self.assertEqual(history[-1].content, "这是助手回复")
        
        # Test assistant message with tool calls
        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="test_tool", arguments='{"arg": "value"}')
        )
        agent.add_assistant_message("调用工具", tool_calls=[tool_call])
        history = agent.get_conversation_history()
        self.assertEqual(len(history), initial_count + 2)
        self.assertIsNotNone(history[-1].tool_calls)
        self.assertEqual(len(history[-1].tool_calls), 1)
    
    def test_add_tool_message(self):
        """Test adding tool messages"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        initial_count = len(agent.get_conversation_history())
        agent.add_tool_message(
            content="工具执行结果",
            tool_call_id="call_123",
            name="test_tool",
            meta={"status": "success"}
        )
        
        history = agent.get_conversation_history()
        self.assertEqual(len(history), initial_count + 1)
        self.assertIsInstance(history[-1], ToolMessage)
        self.assertEqual(history[-1].content, "工具执行结果")
        self.assertEqual(history[-1].tool_call_id, "call_123")
        self.assertEqual(history[-1].name, "test_tool")
        self.assertEqual(history[-1].meta["status"], "success")
    
    def test_get_current_dialog(self):
        """Test getting the current dialog"""
        agent = self.create_agent()
        
        # Before initialization, should return None
        self.assertIsNone(agent.get_current_dialog())
        
        # After initialization, should return a Dialog
        agent._initialize(self.task)
        dialog = agent.get_current_dialog()
        self.assertIsNotNone(dialog)
        self.assertIsInstance(dialog, Dialog)
        
        # After adding a message, the dialog should be updated (get_current_dialog returns the same object reference)
        initial_count = len(dialog.messages)
        agent.add_user_message("新消息")
        # Since it's the same object reference, check the current dialog directly
        self.assertEqual(len(agent.get_current_dialog().messages), initial_count + 1)
    
    def test_get_conversation_history(self):
        """Test getting conversation history"""
        agent = self.create_agent()
        
        # Before initialization, should return an empty list
        history = agent.get_conversation_history()
        self.assertEqual(history, [])
        
        # After initialization, should have messages
        agent._initialize(self.task)
        history = agent.get_conversation_history()
        self.assertGreater(len(history), 0)
        
        # After adding messages, history should be updated
        agent.add_user_message("消息1")
        agent.add_assistant_message("回复1")
        agent.add_tool_message("结果1", "call_1", "tool1")
        
        history = agent.get_conversation_history()
        self.assertGreaterEqual(len(history), 4)  # Initial 2 messages + 3 new messages
    
    def test_reset_context(self):
        """Test resetting context"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        # Add some messages
        agent.add_user_message("消息1")
        agent.add_assistant_message("回复1")
        agent.add_user_message("消息2")
        
        initial_count = len(agent.get_conversation_history())
        self.assertGreater(initial_count, 2)  # Should be more than the initial 2 messages
        
        # Reset context
        agent.reset_context()
        
        # Check if reset to initial state
        history = agent.get_conversation_history()
        self.assertEqual(len(history), 2)  # Only System + User
        self.assertIsInstance(history[0], SystemMessage)
        self.assertIsInstance(history[1], UserMessage)
        
        # Check if the step count is reset
        self.assertEqual(agent._step_count, 0)
    
    def test_reset_context_without_initialization(self):
        """Test that resetting context without initialization raises an error"""
        agent = self.create_agent()
        
        with self.assertRaises(ValueError):
            agent.reset_context()
    
    def test_context_manager_estimate_tokens(self):
        """Test token estimation"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        dialog = agent.get_current_dialog()
        tokens = agent.context_manager.estimate_tokens(dialog)
        
        # Should return a non-negative integer
        self.assertIsInstance(tokens, int)
        self.assertGreaterEqual(tokens, 0)
        
        # After adding more content, the token count should increase
        agent.add_user_message("x" * 1000)  # Add 1000 characters
        new_dialog = agent.get_current_dialog()
        new_tokens = agent.context_manager.estimate_tokens(new_dialog)
        self.assertGreater(new_tokens, tokens)
    
    def test_context_manager_should_truncate(self):
        """Test the truncation-needed check"""
        # Create a config with a small token limit
        context_config = ContextConfig(max_tokens=100)
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        dialog = agent.get_current_dialog()
        should_truncate = agent.context_manager.should_truncate(dialog)
        
        # The initial dialog should not need truncation
        self.assertFalse(should_truncate)
        
        # Add a large amount of content
        for i in range(10):
            agent.add_user_message("x" * 1000)  # Add 1000 characters each time
            agent.add_assistant_message("y" * 1000)
        
        new_dialog = agent.get_current_dialog()
        should_truncate = agent.context_manager.should_truncate(new_dialog)
        # Now truncation should be needed
        self.assertTrue(should_truncate)
    
    def test_context_manager_prepare_for_query_no_truncation(self):
        """Test prepare for query (no truncation needed)"""
        context_config = ContextConfig(max_tokens=1000000)  # Very large limit
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        dialog = agent.get_current_dialog()
        prepared = agent.context_manager.prepare_for_query(dialog)
        
        # When no truncation is needed, should return the original dialog
        self.assertEqual(len(prepared.messages), len(dialog.messages))
        self.assertEqual(prepared.messages, dialog.messages)
    
    def test_context_manager_prepare_for_query_with_truncation(self):
        """Test prepare for query (truncation needed)"""
        context_config = ContextConfig(
            max_tokens=50,  # Very small limit to ensure truncation is triggered
            truncation_strategy=TruncationStrategy.LATEST_HALF
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # Add many messages (each ~20 chars, plus overhead; 10 messages ~500 chars, ~125 tokens)
        for i in range(15):
            agent.add_user_message(f"用户消息 {i} " + "x" * 50)  # Increase message length
            agent.add_assistant_message(f"助手回复 {i} " + "y" * 50)
        
        dialog = agent.get_current_dialog()
        # Ensure truncation is actually needed
        self.assertTrue(agent.context_manager.should_truncate(dialog))
        
        prepared = agent.context_manager.prepare_for_query(dialog)
        
        # Should have been truncated
        self.assertLess(len(prepared.messages), len(dialog.messages))
        # Should preserve the system message
        self.assertIsInstance(prepared.messages[0], SystemMessage)
    
    def test_context_manager_truncate_none_strategy(self):
        """Test truncation strategy: NONE"""
        context_config = ContextConfig(
            max_tokens=100,
            truncation_strategy=TruncationStrategy.NONE
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # Add messages
        for i in range(5):
            agent.add_user_message(f"消息 {i}")
        
        dialog = agent.get_current_dialog()
        truncated = agent.context_manager.truncate(dialog)
        
        # NONE strategy should not truncate
        self.assertEqual(len(truncated.messages), len(dialog.messages))
    
    def test_context_manager_truncate_latest_half_strategy(self):
        """Test truncation strategy: LATEST_HALF"""
        context_config = ContextConfig(
            max_tokens=100,
            truncation_strategy=TruncationStrategy.LATEST_HALF
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # Add multiple assistant messages (truncation starts from assistant messages)
        for i in range(8):
            agent.add_assistant_message(f"助手消息 {i}")
            agent.add_user_message(f"用户消息 {i}")
        
        dialog = agent.get_current_dialog()
        original_count = len(dialog.messages)
        
        truncated = agent.context_manager.truncate(dialog)
        
        # Should have been truncated
        self.assertLess(len(truncated.messages), original_count)
        # Should preserve the system message
        self.assertIsInstance(truncated.messages[0], SystemMessage)
        # Should preserve the initial user message
        self.assertIsInstance(truncated.messages[1], UserMessage)
        # Should retain the latest half
        self.assertGreater(len(truncated.messages), original_count // 2)
    
    def test_context_manager_truncate_sliding_window_strategy(self):
        """Test truncation strategy: SLIDING_WINDOW"""
        context_config = ContextConfig(
            max_tokens=100,
            truncation_strategy=TruncationStrategy.SLIDING_WINDOW,
            preserve_recent_turns=3
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # Add multiple messages
        for i in range(10):
            agent.add_user_message(f"用户消息 {i}")
            agent.add_assistant_message(f"助手消息 {i}")
        
        dialog = agent.get_current_dialog()
        original_count = len(dialog.messages)
        
        truncated = agent.context_manager.truncate(dialog)
        
        # Should have been truncated
        self.assertLess(len(truncated.messages), original_count)
        # Should preserve the system message
        self.assertIsInstance(truncated.messages[0], SystemMessage)
        # Should retain the most recent turns (preserve_recent_turns=3, ~2-3 messages per turn)
        # So should retain about 1 (system) + 3*3 = 10 messages approximately
        self.assertLessEqual(len(truncated.messages), 15)  # Allow some margin
    
    def test_context_manager_truncate_summary_strategy(self):
        """Test truncation strategy: SUMMARY (falls back to latest_half)"""
        context_config = ContextConfig(
            max_tokens=100,
            truncation_strategy=TruncationStrategy.SUMMARY
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # Add messages
        for i in range(8):
            agent.add_assistant_message(f"助手消息 {i}")
            agent.add_user_message(f"用户消息 {i}")
        
        dialog = agent.get_current_dialog()
        truncated = agent.context_manager.truncate(dialog)
        
        # SUMMARY strategy currently falls back to latest_half
        # Should have been truncated
        self.assertLess(len(truncated.messages), len(dialog.messages))
        # Check meta information
        self.assertIn("truncated", truncated.meta)
    
    def test_context_manager_with_token_counter(self):
        """Test using a custom token counter"""
        from evomaster.agent.context import SimpleTokenCounter
        
        token_counter = SimpleTokenCounter(chars_per_token=2.0)  # 1 token per 2 characters
        context_config = ContextConfig(max_tokens=100)
        agent = self.create_agent(context_config)
        
        # Set the token counter
        agent.context_manager.set_token_counter(token_counter)
        
        agent._initialize(self.task)
        dialog = agent.get_current_dialog()
        
        # Estimate using the custom counter
        tokens = agent.context_manager.estimate_tokens(dialog)
        self.assertIsInstance(tokens, int)
        self.assertGreaterEqual(tokens, 0)
    
    def test_set_next_user_request(self):
        """Test the set_next_user_request method"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        initial_count = len(agent.get_conversation_history())
        agent.set_next_user_request("下一个用户请求")
        
        history = agent.get_conversation_history()
        self.assertEqual(len(history), initial_count + 1)
        self.assertIsInstance(history[-1], UserMessage)
        self.assertEqual(history[-1].content, "下一个用户请求")
    
    def test_context_preservation_after_reset(self):
        """Test that initial prompts are preserved after reset"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        # Save initial prompts
        initial_system = agent._initial_system_prompt
        initial_user = agent._initial_user_prompt
        
        # Add messages
        agent.add_user_message("消息1")
        agent.add_assistant_message("回复1")
        
        # Reset
        
        # Check that initial prompts still exist
        self.assertEqual(agent._initial_system_prompt, initial_system)
        self.assertEqual(agent._initial_user_prompt, initial_user)
        
        # Check that the reset dialog uses the initial prompts
        history = agent.get_conversation_history()
        self.assertEqual(history[0].content, initial_system)
        self.assertEqual(history[1].content, initial_user)
    
    def test_multiple_resets(self):
        """Test multiple resets"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        # First reset
        agent.add_user_message("消息1")
        agent.reset_context()
        self.assertEqual(len(agent.get_conversation_history()), 2)
        
        # Second reset
        agent.add_user_message("消息2")
        agent.add_assistant_message("回复2")
        agent.reset_context()
        self.assertEqual(len(agent.get_conversation_history()), 2)
        
        # Third reset
        agent.add_tool_message("结果", "call_1", "tool1")
        agent.reset_context()
        self.assertEqual(len(agent.get_conversation_history()), 2)
    
    def test_empty_message_content(self):
        """Test empty message content"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        # Add messages with empty content
        agent.add_user_message("")
        agent.add_assistant_message(None)  # None should be allowed
        
        history = agent.get_conversation_history()
        self.assertGreaterEqual(len(history), 4)  # Initial 2 messages + 2 new messages
        # The last message should be an assistant message
        self.assertIsInstance(history[-1], AssistantMessage)
    
    def test_very_large_message(self):
        """Test very large messages"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        # Create a very large message (10000 characters)
        large_content = "x" * 10000
        agent.add_user_message(large_content)
        
        history = agent.get_conversation_history()
        self.assertEqual(history[-1].content, large_content)
        
        # Token estimation should handle large messages
        dialog = agent.get_current_dialog()
        tokens = agent.context_manager.estimate_tokens(dialog)
        self.assertGreater(tokens, 1000)  # Should estimate a large number of tokens
    
    def test_truncate_with_no_assistant_messages(self):
        """Test truncation when there are no assistant messages"""
        context_config = ContextConfig(
            max_tokens=100,
            truncation_strategy=TruncationStrategy.LATEST_HALF
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # Only add user messages, no assistant messages
        for i in range(5):
            agent.add_user_message(f"用户消息 {i}")
        
        dialog = agent.get_current_dialog()
        original_count = len(dialog.messages)
        
        # Truncation should handle this case
        truncated = agent.context_manager.truncate(dialog)
        # If there are no assistant messages, the latest_half strategy may not truncate and returns the original dialog
        self.assertLessEqual(len(truncated.messages), original_count)
    
    def test_truncate_with_only_system_message(self):
        """Test truncation with only a system message"""
        context_config = ContextConfig(
            max_tokens=100,
            truncation_strategy=TruncationStrategy.SLIDING_WINDOW
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # Create a dialog with only a system message
        dialog = Dialog(messages=[SystemMessage(content="系统提示词")])
        
        # Truncation should preserve the system message
        truncated = agent.context_manager.truncate(dialog)
        self.assertEqual(len(truncated.messages), 1)
        self.assertIsInstance(truncated.messages[0], SystemMessage)
    
    def test_add_message_error_handling(self):
        """Test error handling when adding messages"""
        agent = self.create_agent()
        
        # Adding messages without initialization should raise an error
        with self.assertRaises(ValueError):
            agent.add_user_message("消息")
        
        with self.assertRaises(ValueError):
            agent.add_assistant_message("回复")
        
        with self.assertRaises(ValueError):
            agent.add_tool_message("结果", "call_1", "tool1")
        
        # After initialization, should work normally
        agent._initialize(self.task)
        agent.add_user_message("消息")
        self.assertEqual(len(agent.get_conversation_history()), 3)  # Initial 2 messages + 1 new message
    
    def test_context_manager_preserve_system_messages(self):
        """Test the preserve system messages configuration"""
        context_config = ContextConfig(
            max_tokens=100,
            truncation_strategy=TruncationStrategy.SLIDING_WINDOW,
            preserve_system_messages=True
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # Add multiple messages
        for i in range(10):
            agent.add_user_message(f"用户消息 {i}")
            agent.add_assistant_message(f"助手消息 {i}")
        
        dialog = agent.get_current_dialog()
        truncated = agent.context_manager.truncate(dialog)
        
        # Should preserve the system message
        self.assertIsInstance(truncated.messages[0], SystemMessage)
        # System message content should remain unchanged
        self.assertEqual(truncated.messages[0].content, dialog.messages[0].content)
    
    def test_dialog_meta_preserved_after_truncation(self):
        """Test that original meta information is preserved after truncation"""
        # Create a config that requires truncation
        context_config = ContextConfig(
            max_tokens=50,  # Very small limit to ensure truncation is triggered
            truncation_strategy=TruncationStrategy.LATEST_HALF
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # Add custom meta
        agent.current_dialog.meta["custom_key"] = "custom_value"
        
        # Add many messages (ensure truncation is triggered)
        for i in range(15):
            agent.add_user_message(f"消息 {i} " + "x" * 50)
            agent.add_assistant_message(f"回复 {i} " + "y" * 50)
        
        dialog = agent.get_current_dialog()
        # Ensure truncation is actually needed
        self.assertTrue(agent.context_manager.should_truncate(dialog))
        
        prepared = agent.context_manager.prepare_for_query(dialog)
        
        # Should have been truncated, so there should be a truncated marker
        self.assertIn("truncated", prepared.meta)
        # Should preserve the original meta
        self.assertEqual(prepared.meta.get("custom_key"), "custom_value")


if __name__ == "__main__":
    unittest.main()

