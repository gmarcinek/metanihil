import luigi
from pathlib import Path

from pipeline.writer.tasks.base_writer_task import BaseWriterTask
from writer.writer_service import WriterService
from writer.models import ChunkStatus
from pipeline.writer.config_loader import ConfigLoader


class EmbedTOCTask(BaseWriterTask):
    toc_path = luigi.Parameter()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toc_name = Path(self.toc_path).stem
        self.config = ConfigLoader()
        self.task_config = self.config.get_task_config('EmbedTOCTask')
        self.output_dir = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/embed_toc"
    
    @property 
    def task_name(self) -> str:
        return "embed_toc"
    
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
            # Initialize WriterService with embedding model from config
            embedding_model = self.task_config.get('model', 'text-embedding-3-small')
            writer_service = WriterService(embedding_model=embedding_model)
            
            # Get all NOT_STARTED chunks (TOC entries)
            chunks_to_embed = writer_service.get_chunks_by_status(ChunkStatus.NOT_STARTED)
            
            if not chunks_to_embed:
                print("✅ No chunks to embed")
                with open(self.output().path, 'w', encoding='utf-8') as f:
                    f.write("No chunks to embed")
                self._persist_task_progress("GLOBAL", "EmbedTOCTask", "COMPLETED")
                return
            
            # Process each chunk
            embedded_count = 0
            for chunk in chunks_to_embed:
                self._persist_task_progress(chunk.hierarchical_id, "EmbedTOCTask", "IN_PROGRESS")
                
                try:
                    # Generate embedding for TOC entry (title only, no content yet)
                    embedding = writer_service.embedder.embed_chunk(chunk, use_content=False)
                    
                    if embedding is not None:
                        # Update chunk with embedding
                        chunk.embedding = embedding
                        chunk.update_timestamp()
                        
                        # Add to FAISS and save
                        writer_service.faiss_manager.add_chunk_embedding(chunk, embedding)
                        writer_service.persistence.save_chunk(chunk)
                        
                        # Update in-memory cache
                        writer_service.chunks[chunk.id] = chunk
                        
                        embedded_count += 1
                        
                        self._persist_task_progress(chunk.hierarchical_id, "EmbedTOCTask", "COMPLETED")
                        print(f"✅ Embedded {chunk.hierarchical_id}: {chunk.title}")
                    else:
                        self._persist_task_progress(chunk.hierarchical_id, "EmbedTOCTask", "FAILED")
                        print(f"❌ Failed to embed {chunk.hierarchical_id}")
                
                except Exception as e:
                    self._persist_task_progress(chunk.hierarchical_id, "EmbedTOCTask", "FAILED")
                    print(f"❌ Error embedding {chunk.hierarchical_id}: {e}")
            
            # Save FAISS index after all embeddings
            writer_service.faiss_manager.save_index()
            
            # Save summary
            summary_file = f"{self.output_dir}/embedding_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Embedded {embedded_count}/{len(chunks_to_embed)} TOC chunks\n")
                f.write(f"Embedding model: {embedding_model}\n")
                f.write(f"FAISS vectors: {writer_service.faiss_manager.get_index_stats().get('total_vectors', 0)}\n")
            
            # Create completion flag
            with open(self.output().path, 'w', encoding='utf-8') as f:
                f.write(f"Completed embedding {embedded_count} TOC chunks")
            
            # Persist task completion
            self._persist_task_progress("GLOBAL", "EmbedTOCTask", "COMPLETED")
            
            print(f"✅ Embedded {embedded_count} TOC chunks and saved to FAISS")
            
        except Exception as e:
            self._persist_task_progress("GLOBAL", "EmbedTOCTask", "FAILED")
            raise
    
    def _persist_task_progress(self, hierarchical_id: str, task_name: str, status: str):
        """Persist task progress to file"""
        progress_file = self.config.get_progress_file()
        Path(progress_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(progress_file, 'a', encoding='utf-8') as f:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp} | {hierarchical_id} | {task_name} | {status}\n")