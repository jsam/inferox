//! Hugging Face Hub API integration

use crate::{Error, FileInfo, RepoInfo, Result};
use reqwest::Client;

/// Hugging Face API client
pub struct HfApiClient {
    client: Client,
    endpoint: String,
    token: Option<String>,
}

impl HfApiClient {
    /// Create a new API client
    pub fn new(endpoint: String, token: Option<String>) -> Result<Self> {
        let client = Client::builder().user_agent("hf-xet-rs/0.1").build()?;

        Ok(Self {
            client,
            endpoint,
            token,
        })
    }

    /// Get repository information
    pub async fn repo_info(&self, repo_id: &str, revision: Option<&str>) -> Result<RepoInfo> {
        let url = if let Some(rev) = revision {
            format!("{}/api/models/{}/revision/{}", self.endpoint, repo_id, rev)
        } else {
            format!("{}/api/models/{}", self.endpoint, repo_id)
        };

        let mut req = self.client.get(&url);

        if let Some(token) = &self.token {
            req = req.header("Authorization", format!("Bearer {}", token));
        }

        let resp = req.send().await?;

        if resp.status() == 404 {
            return Err(Error::RepoNotFound(repo_id.to_string()));
        }

        let info: RepoInfo = resp.error_for_status()?.json().await?;
        Ok(info)
    }

    /// List files in repository
    pub async fn list_files(&self, repo_id: &str, revision: Option<&str>) -> Result<Vec<FileInfo>> {
        let info = self.repo_info(repo_id, revision).await?;
        Ok(info.siblings)
    }

    /// Download file metadata
    pub async fn file_info(
        &self,
        repo_id: &str,
        filename: &str,
        revision: Option<&str>,
    ) -> Result<FileInfo> {
        let files = self.list_files(repo_id, revision).await?;

        files
            .into_iter()
            .find(|f| f.path == filename)
            .ok_or_else(|| Error::FileNotFound(filename.to_string()))
    }

    /// Get download URL for a file
    pub fn download_url(&self, repo_id: &str, filename: &str, revision: Option<&str>) -> String {
        let rev = revision.unwrap_or("main");
        format!("{}/{}/resolve/{}/{}", self.endpoint, repo_id, rev, filename)
    }

    /// Download file directly (for non-LFS files)
    pub async fn download_file(
        &self,
        repo_id: &str,
        filename: &str,
        revision: Option<&str>,
    ) -> Result<bytes::Bytes> {
        let url = self.download_url(repo_id, filename, revision);

        let mut req = self.client.get(&url);

        if let Some(token) = &self.token {
            req = req.header("Authorization", format!("Bearer {}", token));
        }

        let resp = req.send().await?;

        if resp.status() == 404 {
            return Err(Error::FileNotFound(filename.to_string()));
        }

        let bytes = resp.error_for_status()?.bytes().await?;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires network access
    async fn test_repo_info() {
        let client = HfApiClient::new("https://huggingface.co".to_string(), None).unwrap();

        let info = client
            .repo_info("hf-internal-testing/tiny-random-bert", None)
            .await;
        assert!(info.is_ok());
    }
}
