# services/generative_agent/infrastructure/prompt_manager.py
import os
from pathlib import Path
import yaml
from typing import Dict, Any, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..domain.interfaces import PromptManager
from ..domain.models import PromptMetadata, PromptChain, AnalysisType
from ..domain.exceptions import PromptError
from .memo_manager import managed_memoize

class FileSystemPromptManager(PromptManager):
   def __init__(self, 
                base_path: Path,
                config_path: Optional[Path] = None,
                executor: Optional[ThreadPoolExecutor] = None):
       self.base_path = Path(base_path)
       self.config_path = Path(config_path) if config_path else base_path
       self.executor = executor or ThreadPoolExecutor(max_workers=2)
       self.chain_configs: Dict[str, Dict[str, Any]] = {}
       self._load_chain_configs()

   def _load_chain_configs(self):
       try:
           config_file = self.config_path / "chain_configs.yaml"
           if config_file.exists():
               with open(config_file, 'r') as f:
                   self.chain_configs = yaml.safe_load(f)
       except Exception as e:
           raise PromptError(f"Failed to load chain configurations: {str(e)}")

   @managed_memoize(cache_name="prompt_loading", ttl=3600)
   async def load_prompt(self, name: str) -> str:
       try:
           prompt_path = self.base_path / name
           if not prompt_path.exists():
               raise PromptError(f"Prompt file not found: {name}")
           
           loop = asyncio.get_running_loop()
           content = await loop.run_in_executor(
               self.executor,
               self._read_prompt_file,
               prompt_path
           )
           return content
           
       except Exception as e:
           raise PromptError(f"Failed to load prompt {name}: {str(e)}")

   def _read_prompt_file(self, path: Path) -> str:
       with open(path, 'r') as f:
           return f.read().strip()

   @managed_memoize(cache_name="prompt_metadata", ttl=3600)
   async def get_prompt_metadata(self, name: str) -> PromptMetadata:
       try:
           metadata_path = self.base_path / f"{name}.yaml"
           if not metadata_path.exists():
               return self._create_default_metadata(name)
           
           loop = asyncio.get_running_loop()
           metadata = await loop.run_in_executor(
               self.executor,
               self._read_metadata_file,
               metadata_path
           )
           return metadata
           
       except Exception as e:
           raise PromptError(f"Failed to load metadata for {name}: {str(e)}")

   def _read_metadata_file(self, path: Path) -> PromptMetadata:
       with open(path, 'r') as f:
           data = yaml.safe_load(f)
           return PromptMetadata(
               name=data.get('name', ''),
               description=data.get('description', ''),
               version=data.get('version', '1.0'),
               required_context=data.get('required_context', []),
               optional_context=data.get('optional_context', []),
               chain_compatible=data.get('chain_compatible', [])
           )

   def _create_default_metadata(self, name: str) -> PromptMetadata:
       return PromptMetadata(
           name=name,
           description=f"Prompt for {name}",
           version="1.0",
           required_context=[],
           optional_context=[],
           chain_compatible=[]
       )

   @managed_memoize(cache_name="prompt_chains", ttl=1800)
   async def create_chain(
       self,
       analysis_type: AnalysisType,
       context: Dict[str, Any]
   ) -> PromptChain:
       try:
           chain_config = self.chain_configs.get(analysis_type.value, {})
           if not chain_config:
               raise PromptError(f"No chain configuration for {analysis_type.value}")
           
           prompt_names = chain_config.get('prompts', [])
           prompts = [await self.load_prompt(name) for name in prompt_names]
           
           return PromptChain(
               chain_id=f"{analysis_type.value}_{context.get('request_id', '')}",
               prompts=prompts,
               metadata={'analysis_type': analysis_type.value, 'context': context},
               total_tokens=0  # TODO: Implement token counting
           )
           
       except Exception as e:
           raise PromptError(f"Failed to create prompt chain: {str(e)}")

   def get_available_chains(self) -> List[str]:
       return list(self.chain_configs.keys())
