//! Local caching system for XET chunks

use crate::Result;
use std::fs;
use std::path::{Path, PathBuf};

/// Chunk cache manager
pub struct ChunkCache {
    cache_dir: PathBuf,
    #[allow(dead_code)]
    max_size: u64,
}

impl ChunkCache {
    /// Create a new chunk cache
    pub fn new(cache_dir: PathBuf, max_size: u64) -> Result<Self> {
        fs::create_dir_all(&cache_dir)?;
        Ok(Self {
            cache_dir,
            max_size,
        })
    }

    /// Get chunk from cache
    pub fn get(&self, chunk_id: &str) -> Option<Vec<u8>> {
        let path = self.chunk_path(chunk_id);
        fs::read(&path).ok()
    }

    /// Store chunk in cache
    pub fn put(&self, chunk_id: &str, data: &[u8]) -> Result<()> {
        let path = self.chunk_path(chunk_id);

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(&path, data)?;
        Ok(())
    }

    /// Get current cache size
    pub fn size(&self) -> Result<u64> {
        let mut total_size = 0;
        self.walk_cache(&self.cache_dir, &mut total_size)?;
        Ok(total_size)
    }

    /// Count chunks in cache
    pub fn count(&self) -> Result<usize> {
        let mut count = 0;
        self.count_files(&self.cache_dir, &mut count)?;
        Ok(count)
    }

    /// Evict old entries to maintain size limit
    pub fn evict_lru(&mut self, target_size: u64) -> Result<u64> {
        let current_size = self.size()?;

        if current_size <= target_size {
            return Ok(0);
        }

        let mut entries = Vec::new();
        self.collect_entries(&self.cache_dir, &mut entries)?;

        entries
            .sort_by_key(|(_, metadata)| metadata.accessed().or_else(|_| metadata.modified()).ok());

        let mut freed = 0;
        let to_free = current_size - target_size;

        for (path, metadata) in entries {
            if freed >= to_free {
                break;
            }

            let size: u64 = metadata.len();
            if fs::remove_file(&path).is_ok() {
                freed += size;
            }
        }

        Ok(freed)
    }

    /// Clear all cache
    pub fn clear(&self) -> Result<()> {
        if self.cache_dir.exists() {
            fs::remove_dir_all(&self.cache_dir)?;
            fs::create_dir_all(&self.cache_dir)?;
        }
        Ok(())
    }

    fn chunk_path(&self, chunk_id: &str) -> PathBuf {
        let prefix = if chunk_id.len() >= 2 {
            &chunk_id[..2]
        } else {
            chunk_id
        };

        self.cache_dir
            .join("chunks")
            .join(prefix)
            .join(chunk_id)
            .join("data")
    }

    #[allow(clippy::only_used_in_recursion)]
    fn walk_cache(&self, dir: &Path, total: &mut u64) -> Result<()> {
        if !dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let metadata = entry.metadata()?;

            if metadata.is_dir() {
                self.walk_cache(&entry.path(), total)?;
            } else {
                *total += metadata.len();
            }
        }
        Ok(())
    }

    #[allow(clippy::only_used_in_recursion)]
    fn count_files(&self, dir: &Path, count: &mut usize) -> Result<()> {
        if !dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let metadata = entry.metadata()?;

            if metadata.is_dir() {
                self.count_files(&entry.path(), count)?;
            } else if entry.file_name() == "data" {
                *count += 1;
            }
        }
        Ok(())
    }

    #[allow(clippy::only_used_in_recursion)]
    fn collect_entries(
        &self,
        dir: &Path,
        entries: &mut Vec<(PathBuf, fs::Metadata)>,
    ) -> Result<()> {
        if !dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let metadata = entry.metadata()?;

            if metadata.is_dir() {
                self.collect_entries(&path, entries)?;
            } else if entry.file_name() == "data" {
                entries.push((path, metadata));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_cache_put_get() {
        let dir = tempdir().unwrap();
        let cache = ChunkCache::new(dir.path().to_path_buf(), 1024 * 1024).unwrap();

        let chunk_id = "abcd1234";
        let data = b"test data";

        cache.put(chunk_id, data).unwrap();
        let retrieved = cache.get(chunk_id).unwrap();

        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_cache_clear() {
        let dir = tempdir().unwrap();
        let cache = ChunkCache::new(dir.path().to_path_buf(), 1024 * 1024).unwrap();

        cache.put("chunk1", b"data1").unwrap();
        cache.put("chunk2", b"data2").unwrap();

        assert!(cache.get("chunk1").is_some());

        cache.clear().unwrap();

        assert!(cache.get("chunk1").is_none());
    }
}
