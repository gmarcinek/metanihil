import luigi
from pathlib import Path
import json

from components.structured_task import StructuredTask
from pipeline.writer.database import DatabaseManager
from pipeline.writer.models import ChunkStatus
from pipeline.writer.config_loader import ConfigLoader
from llm import LLMClient


class EmbedTOCTask(StructuredTask):
    toc_path = luigi.Parameter()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toc_name = Path(self.toc_path).stem
        self.config = ConfigLoader()
        self.task_config = self.config.get_task_config('EmbedTOCTask')
        self.output_dir = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/embed_toc"
        self.embeddings_cache = f"{self.output_dir}/{self.config.get_cache_config()['embeddings_filename']}"
    
    def requires(self):
        from .parse_toc_task import ParseTOCTask
        return ParseTOCTask(toc_path=self.toc_path)
    
    def output(self):
        return luigi.LocalTarget(f"{self.output_dir}/completed.flag")
    
    def run(self):
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Persist task start
        self._persist_task_progress("GLOBAL", "EmbedTOCTask", "STARTED")
        
        try:
            # Get all chunks from database
            db = DatabaseManager(self.config.get_database_path())
            chunks = db.get_chunks_by_status(ChunkStatus.NOT_STARTED)
            
            # Initialize LLM client for embeddings with configured model
            llm_client = LLMClient(model=self.task_config['model'])
            
            # Load existing cache
            embeddings_cache = self._load_embeddings_cache()
            
            # Process each chunk
            for chunk in chunks:
                self._persist_task_progress(chunk.hierarchical_id, "EmbedTOCTask", "IN_PROGRESS")
                
                # Create embedding text (hierarchical_id + title)
                embedding_text = f"{chunk.hierarchical_id} {chunk.title}"
                
                # Check cache first
                if embedding_text in embeddings_cache:
                    print(f"📋 Using cached embedding for {chunk.hierarchical_id}")
                    self._persist_task_progress(chunk.hierarchical_id, "EmbedTOCTask", "COMPLETED")
                    continue
                
                # Generate embedding
                embedding = llm_client.get_embedding(embedding_text)
                
                # Cache the embedding
                embeddings_cache[embedding_text] = {
                    "hierarchical_id": chunk.hierarchical_id,
                    "embedding": embedding,
                    "chunk_id": chunk.id
                }
                
                # Save cache after each embedding
                self._save_embeddings_cache(embeddings_cache)
                
                self._persist_task_progress(chunk.hierarchical_id, "EmbedTOCTask", "COMPLETED")
                print(f"✅ Embedded {chunk.hierarchical_id}: {chunk.title}")
            
            # Create completion flag
            with self.output().open('w') as f:
                f.write(f"Completed embedding {len(chunks)} TOC lines")
            
            # Persist task completion
            self._persist_task_progress("GLOBAL", "EmbedTOCTask", "COMPLETED")
            
            print(f"✅ Embedded {len(chunks)} TOC lines and cached")
            
        except Exception as e:
            self._persist_task_progress("GLOBAL", "EmbedTOCTask", "FAILED")
            raise
    
    def _load_embeddings_cache(self) -> dict:
        """Load embeddings cache from file"""
        if Path(self.embeddings_cache).exists():
            with open(self.embeddings_cache, 'r') as f:
                cache = json.load(f)
            print(f"📖 Loaded embeddings cache with {len(cache)} entries from {self.embeddings_cache}")
            return cache
        else:
            print(f"📝 No existing cache found, starting fresh at {self.embeddings_cache}")
            return {}
    
    def _save_embeddings_cache(self, cache: dict):
        """Save embeddings cache to file"""
        with open(self.embeddings_cache, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"💾 Saved embeddings cache with {len(cache)} entries to {self.embeddings_cache}")
    
    def _persist_task_progress(self, hierarchical_id: str, task_name: str, status: str):
        """Persist task progress to file"""
        progress_file = self.config.get_progress_file()
        Path(progress_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(progress_file, 'a') as f:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp} | {hierarchical_id} | {task_name} | {status}\n")