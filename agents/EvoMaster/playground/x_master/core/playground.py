import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from functools import partial

# 确保可以导入evomaster模块
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.core import BasePlayground, register_playground
from evomaster import TaskInstance

from .exp import SolveExp, CritiqueExp, RewriteExp, SelectExp

@register_playground("x_master")
class XMasterPlayground(BasePlayground):
    """X-Master Playground
    
    协调四个Exp类，实现完整的X-Master工作流。
    支持结果有效性验证与自动重试。
    """
    
    def __init__(self, config_dir: Path = None, config_path: Path = None):
        """初始化X-Master Playground
        
        Args:
            config_dir: 配置目录路径，默认为 configs/xmaster/
            config_path: 配置文件完整路径
        """
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "agent" / "x_master"
        
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("solver_agent", "critic_agent", "rewriter_agent", "selector_agent")
        
        # 存储中间结果
        self.solver_results = []
        self.critic_results = []
        self.rewriter_results = []
        self.selector_results = []

        self.mcp_manager = None
        self.max_retries = 3  # 默认值，将在_setup中覆盖
        
    def setup(self) -> None:
        """初始化所有组件
        
        创建四个Agent和对应的Exp实例。
        """
        self.logger.info("Setting up X-Master playground...")
        
        # 1. 创建 Session
        self._setup_session()
        
        # 2. 创建工具注册表
        self._setup_tools()

        # 3. 从配置中获取工作流参数
        self._load_workflow_config()

        # 4. 创建四个组件的Agent
        self._setup_agents()
        
        self.logger.info("X-Master playground setup complete")
    
    def _load_workflow_config(self) -> None:
        """从配置中加载工作流参数
        """
        xmaster_config = getattr(self.config, 'xmaster', {})
        if not xmaster_config:
            xmaster_config = {}

        self.agent_num = xmaster_config.get('agent_num', 1)
        self.max_workers = xmaster_config.get('max_workers', 1)
        self.parallel = xmaster_config.get('parallel', True)
        self.max_retries = xmaster_config.get('max_retries', 3)  # 新增重试次数配置

        self.logger.info(f"Workflow config: agent_num={self.agent_num}, max_workers={self.max_workers}, max_retries={self.max_retries}")

    def _is_valid_result(self, result_dict: Dict[str, Any], key: str) -> bool:
        """检查结果是否有效（非None且非空字符串）
        
        Args:
            result_dict: 实验返回的结果字典
            key: 需要检查的键名
            
        Returns:
            是否有效
        """
        if key not in result_dict:
            return False
        value = result_dict[key]
        if value is None:
            return False
        if isinstance(value, str) and value.strip() == "":
            return False
        # 可根据需要扩展其他类型的验证
        return True

    def _create_exp(self, exp_index, exp_name:str):
        """创建多智能体实验实例"""
        exp = None
        
        if exp_name == "solve":
            solver_agent_copy = self.copy_agent(
                self.agents.solver_agent, 
                new_agent_name=f"solve_exp_{exp_index}"
            ) if self.agents.solver_agent else None
            exp = SolveExp(
                solver_agent=solver_agent_copy,
                config=self.config,
                index=exp_index
            )
        
        elif exp_name == "critique":
            critic_agent_copy = self.copy_agent(
                self.agents.critic_agent, 
                new_agent_name=f"critique_exp_{exp_index}"
            ) if self.agents.critic_agent else None
            exp = CritiqueExp(
                critic_agent=critic_agent_copy,
                config=self.config,
                index=exp_index
            )
        
        elif exp_name == "rewrite":
            rewriter_agent_copy = self.copy_agent(
                self.agents.rewriter_agent, 
                new_agent_name=f"rewrite_exp_{exp_index}"
            ) if self.agents.rewriter_agent else None
            exp = RewriteExp(
                rewriter_agent=rewriter_agent_copy,
                config=self.config,
                index=exp_index
            )
        
        elif exp_name == "select":
            selector_agent_copy = self.copy_agent(
                self.agents.selector_agent, 
                new_agent_name=f"select_exp_{exp_index}"
            ) if self.agents.selector_agent else None
            exp = SelectExp(  
                selector_agent=selector_agent_copy,
                config=self.config,
                index=exp_index
            )
        
        else:
            raise ValueError(f"Unknown exp_name: {exp_name}. Expected one of: solve, critique, rewrite, select")
        
        if exp is None:
            raise RuntimeError(f"Failed to create exp: {exp_name}")
        
        return exp
    
    def _extract_solutions_from_results(self, results: List) -> List[str]:
        """从Exp结果中提取解决方案列表"""
        solutions = []
        #直接查找 solutions_result_{i}
        for result in results:
            index = result['exp_index']
            key = f"solver_result"
            if key in result and result[key] is not None:
                solutions.append(result[key])
                self.logger.info(f"index:{index} 找到 {key}: {result[key][:50]}...")
            elif key in result:
                self.logger.warning(f"index:{index} {key} 的值为 None，跳过")
        self.logger.info(f"最终提取到 {len(solutions)} 个解决方案")
        return solutions
    
    def _extract_corrected_solutions(self, results: List) -> List[str]:
        """从Critic结果中提取修正后的解决方案
        
        Args:
            results: CritiqueExp运行结果
            
        Returns:
            修正后的解决方案列表
        """
        solutions = []
        #直接查找 critic_result_{i}
        for result in results:
            index = result['exp_index']
            key = f"critic_result"
            if key in result and result[key] is not None:
                solutions.append(result[key])
                self.logger.info(f"index:{index} 找到 {key}: {result[key][:50]}...")
            elif key in result:
                self.logger.warning(f"index:{index} {key} 的值为 None，跳过")
        self.logger.info(f"最终提取到 {len(solutions)} 个解决方案")
        return solutions

    def _extract_rewritten_solutions(self, results: List) -> List[str]:
        """从Rewriter结果中提取重写后的解决方案
        
        Args:
            results: RewriteExp运行结果
            
        Returns:
            重写后的解决方案列表
        """
        solutions = []
        #直接查找 rewriter_result_{i}
        for result in results:
            index = result['exp_index']
            key = f"rewriter_result"
            if key in result and result[key] is not None:
                solutions.append(result[key])
                self.logger.info(f"index:{index} 找到 {key}: {result[key][:50]}...")
            elif key in result:
                self.logger.warning(f"index:{index} {key} 的值为 None，跳过")
        self.logger.info(f"最终提取到 {len(solutions)} 个解决方案")
        return solutions
    
    def _extract_selected_solution(self, results: Dict[str, Any]) -> str:
        """从Selector结果中提取选中的解决方案
        
        Args:
            results: SelectExp运行结果
            
        Returns:
            选中的解决方案
        """
        key = f"selector_result"
        solutions = results[key]
        self.logger.info(f"找到 {key}: {results[key][:50]}...")
        return solutions
    
    def _run_with_parallel(self, task_description: str, task_id: str = None):
        """并行执行工作流（带重试机制）"""
        self.logger.info(f"=== Parallel Process ({self.agent_num} agents) ===")
        
        # ---------- Phase 1: Solver (并行，带重试) ----------
        self.logger.info(f"=== Phase 1: Solver (parallel, {self.agent_num} agents) ===")
        
        # 初始化最终结果列表
        final_solutions = [None] * self.agent_num
        # 需要重试的索引列表
        pending_indices = list(range(self.agent_num))
        retry = 0
        
        while retry < self.max_retries and pending_indices:
            tasks = []
            for idx in pending_indices:
                exp = self._create_exp(exp_index=idx, exp_name="solve")
                task_func = partial(
                    exp.run,
                    task_description=task_description,
                    task_id=f"{task_id}_solver" + (f"_retry{retry}" if retry > 0 else "")
                )
                tasks.append((idx, task_func))
            
            # 执行当前批次任务
            current_tasks = [tf for _, tf in tasks]
            results_batch = self.execute_parallel_tasks(current_tasks, max_workers=self.max_workers)
            
            # 更新结果
            new_pending = []
            for (idx, _), result in zip(tasks, results_batch):
                key = "solver_result"
                if self._is_valid_result(result, key):
                    final_solutions[idx] = result[key]
                    self.logger.info(f"index:{idx} Solver 成功")
                else:
                    self.logger.warning(f"index:{idx} Solver 结果无效，将重试")
                    new_pending.append(idx)
            
            pending_indices = new_pending
            retry += 1
            if pending_indices:
                self.logger.warning(f"Solver 阶段仍有 {len(pending_indices)} 个无效结果，第 {retry} 次重试")
        
        if pending_indices:
            raise RuntimeError(f"Solver 阶段在 {self.max_retries} 次重试后仍未能获得所有有效结果，缺失索引: {pending_indices}")
        
        # 转换为标准结果格式
        self.solver_results = [{"exp_index": i, "solver_result": final_solutions[i]} for i in range(self.agent_num)]
        original_solutions = final_solutions
        self.logger.info(f"Solver generated {len(original_solutions)} solutions")

        # ---------- Phase 2: Critic (并行，一一对应，带重试) ----------
        self.logger.info(f"=== Phase 2: Critic (parallel, {self.agent_num} agents) ===")
        
        final_critic_solutions = [None] * self.agent_num
        pending_indices = list(range(self.agent_num))
        retry = 0
        
        while retry < self.max_retries and pending_indices:
            tasks = []
            for idx in pending_indices:
                exp = self._create_exp(exp_index=idx, exp_name="critique")
                task_func = partial(
                    exp.run,
                    task_description=task_description,
                    solution=original_solutions[idx],  # 传入对应的原始解
                    task_id=f"{task_id}_critic" + (f"_retry{retry}" if retry > 0 else "")
                )
                tasks.append((idx, task_func))
            
            current_tasks = [tf for _, tf in tasks]
            results_batch = self.execute_parallel_tasks(current_tasks, max_workers=self.max_workers)
            
            new_pending = []
            for (idx, _), result in zip(tasks, results_batch):
                key = "critic_result"
                if self._is_valid_result(result, key):
                    final_critic_solutions[idx] = result[key]
                    self.logger.info(f"index:{idx} Critic 成功")
                else:
                    self.logger.warning(f"index:{idx} Critic 结果无效，将重试")
                    new_pending.append(idx)
            
            pending_indices = new_pending
            retry += 1
            if pending_indices:
                self.logger.warning(f"Critic 阶段仍有 {len(pending_indices)} 个无效结果，第 {retry} 次重试")
        
        if pending_indices:
            raise RuntimeError(f"Critic 阶段在 {self.max_retries} 次重试后仍未能获得所有有效结果，缺失索引: {pending_indices}")
        
        self.critic_results = [{"exp_index": i, "critic_result": final_critic_solutions[i]} for i in range(self.agent_num)]
        corrected_solutions = final_critic_solutions
        self.logger.info(f"Critic generated {len(corrected_solutions)} corrected solutions")

        # ---------- Phase 3: Rewriter (并行，综合所有，带重试) ----------
        self.logger.info(f"=== Phase 3: Rewriter (parallel, {self.agent_num} agents) ===")
        
        final_rewritten_solutions = [None] * self.agent_num
        pending_indices = list(range(self.agent_num))
        retry = 0
        
        # 重写阶段依赖所有修正后的解决方案，该列表已完整
        all_corrected = corrected_solutions
        
        while retry < self.max_retries and pending_indices:
            tasks = []
            for idx in pending_indices:
                exp = self._create_exp(exp_index=idx, exp_name="rewrite")
                task_func = partial(
                    exp.run,
                    task_description=task_description,
                    solutions=all_corrected,  # 传入所有修正解
                    task_id=f"{task_id}_rewriter" + (f"_retry{retry}" if retry > 0 else "")
                )
                tasks.append((idx, task_func))
            
            current_tasks = [tf for _, tf in tasks]
            results_batch = self.execute_parallel_tasks(current_tasks, max_workers=self.max_workers)
            
            new_pending = []
            for (idx, _), result in zip(tasks, results_batch):
                key = "rewriter_result"
                if self._is_valid_result(result, key):
                    final_rewritten_solutions[idx] = result[key]
                    self.logger.info(f"index:{idx} Rewriter 成功")
                else:
                    self.logger.warning(f"index:{idx} Rewriter 结果无效，将重试")
                    new_pending.append(idx)
            
            pending_indices = new_pending
            retry += 1
            if pending_indices:
                self.logger.warning(f"Rewriter 阶段仍有 {len(pending_indices)} 个无效结果，第 {retry} 次重试")
        
        if pending_indices:
            raise RuntimeError(f"Rewriter 阶段在 {self.max_retries} 次重试后仍未能获得所有有效结果，缺失索引: {pending_indices}")
        
        self.rewriter_results = [{"exp_index": i, "rewriter_result": final_rewritten_solutions[i]} for i in range(self.agent_num)]
        rewritten_solutions = final_rewritten_solutions
        self.logger.info(f"Rewriter generated {len(rewritten_solutions)} rewritten solutions")

        # ---------- Phase 4: Selector (单 Agent，带重试) ----------
        self.logger.info("=== Phase 4: Selector ===")
        
        selected_solution = None
        retry = 0
        while retry < self.max_retries and selected_solution is None:
            selector_exp = self._create_exp(exp_index=0, exp_name="select")
            result = selector_exp.run(
                task_description=task_description,
                solutions=rewritten_solutions,
                task_id=f"{task_id}_selector" + (f"_retry{retry}" if retry > 0 else "")
            )
            key = "selector_result"
            if self._is_valid_result(result, key):
                selected_solution = result[key]
                self.logger.info("Selector 成功")
            else:
                self.logger.warning(f"Selector 结果无效，第 {retry+1} 次重试")
                retry += 1
        
        if selected_solution is None:
            raise RuntimeError(f"Selector 阶段在 {self.max_retries} 次重试后仍未能获得有效结果")
        
        self.selector_results = result  # 保持原始格式
        self.logger.info("Selector completed, best solution selected")

        return original_solutions, corrected_solutions, rewritten_solutions, selected_solution

    def _run_with_serial(self, task_description: str, task_id: str = None):
        """串行执行工作流（带重试机制）"""
        self.logger.info(f"=== Serial Process ({self.agent_num} agents) ===")
        
        # 1. Solver阶段
        self.logger.info(f"=== Phase 1: Solver (serial, {self.agent_num} agents) ===")
        solver_solutions = []
        for i in range(self.agent_num):
            solution = None
            retry = 0
            while retry < self.max_retries and solution is None:
                exp = self._create_exp(exp_index=i, exp_name="solve")
                result = exp.run(
                    task_description=task_description,
                    task_id=f"{task_id}_solver_{i}" + (f"_retry{retry}" if retry > 0 else "")
                )
                key = "solver_result"
                if self._is_valid_result(result, key):
                    solution = result[key]
                    self.logger.info(f"index:{i} Solver 成功")
                else:
                    self.logger.warning(f"index:{i} Solver 结果无效，第 {retry+1} 次重试")
                    retry += 1
            if solution is None:
                raise RuntimeError(f"Solver 阶段 index {i} 在 {self.max_retries} 次重试后仍失败")
            solver_solutions.append(solution)
            self.solver_results.append({"exp_index": i, "solver_result": solution})
        
        original_solutions = solver_solutions
        self.logger.info(f"Solver generated {len(original_solutions)} solutions")

        # 2. Critic阶段
        self.logger.info(f"=== Phase 2: Critic (serial, {self.agent_num} agents) ===")
        critic_solutions = []
        for i in range(self.agent_num):
            solution = None
            retry = 0
            while retry < self.max_retries and solution is None:
                exp = self._create_exp(exp_index=i, exp_name="critique")
                result = exp.run(
                    task_description=task_description,
                    solution=original_solutions[i],
                    task_id=f"{task_id}_critic_{i}" + (f"_retry{retry}" if retry > 0 else "")
                )
                key = "critic_result"
                if self._is_valid_result(result, key):
                    solution = result[key]
                    self.logger.info(f"index:{i} Critic 成功")
                else:
                    self.logger.warning(f"index:{i} Critic 结果无效，第 {retry+1} 次重试")
                    retry += 1
            if solution is None:
                raise RuntimeError(f"Critic 阶段 index {i} 在 {self.max_retries} 次重试后仍失败")
            critic_solutions.append(solution)
            self.critic_results.append({"exp_index": i, "critic_result": solution})
        
        corrected_solutions = critic_solutions
        self.logger.info(f"Critic generated {len(corrected_solutions)} corrected solutions")

        # 3. Rewriter阶段
        self.logger.info(f"=== Phase 3: Rewriter (serial, {self.agent_num} agents) ===")
        rewritten_solutions = []
        for i in range(self.agent_num):
            solution = None
            retry = 0
            while retry < self.max_retries and solution is None:
                exp = self._create_exp(exp_index=i, exp_name="rewrite")
                result = exp.run(
                    task_description=task_description,
                    solutions=corrected_solutions,
                    task_id=f"{task_id}_rewriter_{i}" + (f"_retry{retry}" if retry > 0 else "")
                )
                key = "rewriter_result"
                if self._is_valid_result(result, key):
                    solution = result[key]
                    self.logger.info(f"index:{i} Rewriter 成功")
                else:
                    self.logger.warning(f"index:{i} Rewriter 结果无效，第 {retry+1} 次重试")
                    retry += 1
            if solution is None:
                raise RuntimeError(f"Rewriter 阶段 index {i} 在 {self.max_retries} 次重试后仍失败")
            rewritten_solutions.append(solution)
            self.rewriter_results.append({"exp_index": i, "rewriter_result": solution})
        
        self.logger.info(f"Rewriter generated {len(rewritten_solutions)} rewritten solutions")

        # 4. Selector阶段
        self.logger.info("=== Phase 4: Selector ===")
        selected_solution = None
        retry = 0
        while retry < self.max_retries and selected_solution is None:
            exp = self._create_exp(exp_index=0, exp_name="select")
            result = exp.run(
                task_description=task_description,
                solutions=rewritten_solutions,
                task_id=f"{task_id}_selector" + (f"_retry{retry}" if retry > 0 else "")
            )
            key = "selector_result"
            if self._is_valid_result(result, key):
                selected_solution = result[key]
                self.logger.info("Selector 成功")
            else:
                self.logger.warning(f"Selector 结果无效，第 {retry+1} 次重试")
                retry += 1
        
        if selected_solution is None:
            raise RuntimeError(f"Selector 阶段在 {self.max_retries} 次重试后仍未能获得有效结果")
        
        self.selector_results = result
        self.logger.info("Selector completed, best solution selected")

        return original_solutions, corrected_solutions, rewritten_solutions, selected_solution
    
    def run_xmaster_workflow(self, task_description: str, task_id: str = None) -> Dict[str, Any]:
        """运行完整的X-Master工作流
        
        Args:
            task_description: 任务描述
            task_id: 任务ID（用于批量处理）
            
        Returns:
            完整的X-Master工作流结果
        """
        if not task_id:
            task_id = "xmaster_task_001"
        
        self.logger.info(f"Starting X-Master workflow for task: {task_id}")
        self.logger.info(f"Task description: {task_description[:100]}...")
        
        # 是否并行处理
        if self.parallel:
            original_solutions, corrected_solutions, rewritten_solutions, selected_solution = self._run_with_parallel(task_description, task_id)
        else:
            original_solutions, corrected_solutions, rewritten_solutions, selected_solution = self._run_with_serial(task_description, task_id)

        # 构建最终结果
        final_result = {
            "status": "completed",
            "task_id": task_id,
            "task_description": task_description,
            "final_solution": selected_solution,
            "phase_results": {
                "solver": original_solutions,
                "critic": corrected_solutions,
                "rewriter": rewritten_solutions,
                "selector": selected_solution
            },
            "solutions_summary": {
                "original_count": len(original_solutions),
                "corrected_count": len(corrected_solutions),
                "rewritten_count": len(rewritten_solutions)
            }
        }
        
        self.logger.info("X-Master workflow completed successfully")
        
        return final_result
    
    def run(self, task_description: str, output_file: str | None = None) -> Dict[str, Any]:
        """运行X-Master工作流（覆盖基类方法）

        Args:
            task_description: 任务描述
            output_file: 结果保存文件

        Returns:
            运行结果
        """
        try:
            self.setup()

            # 设置 trajectory 文件路径（使用基类方法，统一目录结构）
            self._setup_trajectory_file(output_file)

            # 运行完整的X-Master工作流
            task_id = getattr(self, 'task_id', None)
            final_result = self.run_xmaster_workflow(task_description, task_id)

            return final_result

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """清理资源
        
        覆盖基类方法，清理所有Agent和Exp。
        """
        # 清理基类资源
        super().cleanup()
             
        # 清空结果
        self.solver_results = None
        self.critic_results = None
        self.rewriter_results = None
        self.selector_results = None
        
        self.logger.debug("X-Master resources cleaned up")