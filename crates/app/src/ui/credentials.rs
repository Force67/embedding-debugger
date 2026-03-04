use embedding_inference::ProviderKind;
use keyring::Entry;

const SERVICE_NAME: &str = "embedding-debugger";

fn account_name(provider: ProviderKind) -> &'static str {
    match provider {
        ProviderKind::OpenRouter => "provider:openrouter",
        ProviderKind::OpenAI => "provider:openai",
        ProviderKind::Azure => "provider:azure",
    }
}

/// Load a provider-specific API key from the OS keyring.
pub fn load_api_key(provider: ProviderKind) -> Option<String> {
    let entry = Entry::new(SERVICE_NAME, account_name(provider)).ok()?;
    let key = entry.get_password().ok()?;
    if key.trim().is_empty() {
        None
    } else {
        Some(key)
    }
}

/// Persist or clear a provider-specific API key in the OS keyring.
pub fn store_api_key(provider: ProviderKind, key: &str) -> Result<(), keyring::Error> {
    let entry = Entry::new(SERVICE_NAME, account_name(provider))?;

    if key.trim().is_empty() {
        // Ignore "no entry found" or backend-specific delete failures.
        let _ = entry.delete_password();
        return Ok(());
    }

    entry.set_password(key)
}
