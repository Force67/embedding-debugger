use iced::widget::{button, column, container, pick_list, text, text_input, Space};
use iced::{Element, Length, Padding};

use embedding_core::TokenCollection;

use super::app::Message;

/// State for the sidebar UI controls.
#[derive(Debug, Clone, Default)]
pub struct SidebarState {
    pub provider_input: String,
    pub model_input: String,
    pub api_key_input: String,
    pub dimensions_input: String,
    /// Azure: full endpoint URL
    pub endpoint_input: String,
    /// Azure: deployment name
    pub deployment_input: String,
    /// Comma- or newline-separated tokens typed/pasted directly.
    pub tokens_text: String,
}

const PROVIDERS: &[&str] = &["OpenRouter", "OpenAI", "Azure"];

pub fn view<'a>(
    state: &'a SidebarState,
    tokens: &'a Option<TokenCollection>,
    loading: bool,
) -> Element<'a, Message> {
    let is_azure = state.provider_input == "Azure";

    let provider_picker = pick_list(
        PROVIDERS.to_vec(),
        Some(if state.provider_input.is_empty() {
            "OpenRouter"
        } else {
            state.provider_input.as_str()
        }),
        |s| Message::ProviderChanged(s.to_string()),
    )
    .width(Length::Fill);

    let api_key_input = text_input("API Key", &state.api_key_input)
        .on_input(Message::ApiKeyChanged)
        .secure(true)
        .width(Length::Fill);

    let dims_input = text_input("Dimensions (optional)", &state.dimensions_input)
        .on_input(Message::DimensionsChanged)
        .width(Length::Fill);

    let load_button = button(text("📂 Load from file"))
        .on_press(Message::LoadTokensPressed)
        .width(Length::Fill);

    let tokens_input = text_input("dog, cat, apple, king… (comma/newline)", &state.tokens_text)
        .on_input(Message::TokensTextChanged)
        .width(Length::Fill);

    let use_tokens_button = if state.tokens_text.trim().is_empty() {
        button(text("Use typed tokens")).width(Length::Fill)
    } else {
        button(text("Use typed tokens"))
            .on_press(Message::UseTypedTokens)
            .width(Length::Fill)
    };

    let token_info = match tokens {
        Some(collection) => {
            text(format!("{}: {} tokens", collection.name, collection.len()))
        }
        None => text("No tokens loaded"),
    };

    let generate_button = if loading {
        button(text("Generating...")).width(Length::Fill)
    } else {
        button(text("Generate Embeddings"))
            .on_press(Message::GeneratePressed)
            .width(Length::Fill)
    };

    // Build the provider-specific section of the form.
    let provider_fields: Element<'a, Message> = if is_azure {
        column![
            text("Endpoint").size(14),
            text_input(
                "https://resource.cognitiveservices.azure.com/openai/v1/",
                &state.endpoint_input
            )
            .on_input(Message::EndpointChanged)
            .width(Length::Fill),
            Space::with_height(8),
            text("Deployment Name").size(14),
            text_input("text-embedding-3-large", &state.deployment_input)
                .on_input(Message::DeploymentChanged)
                .width(Length::Fill),
            Space::with_height(8),
            text("API Key").size(14),
            api_key_input,
        ]
        .spacing(4)
        .into()
    } else {
        let model_input =
            text_input("Model (e.g. openai/text-embedding-3-small)", &state.model_input)
                .on_input(Message::ModelChanged)
                .width(Length::Fill);
        column![
            text("Model").size(14),
            model_input,
            Space::with_height(8),
            text("API Key").size(14),
            api_key_input,
        ]
        .spacing(4)
        .into()
    };

    let content = column![
        text("Provider").size(14),
        provider_picker,
        Space::with_height(8),
        provider_fields,
        Space::with_height(8),
        text("Dimensions").size(14),
        dims_input,
        Space::with_height(16),
        text("Tokens").size(16),
        tokens_input,
        use_tokens_button,
        load_button,
        token_info,
        Space::with_height(16),
        generate_button,
    ]
    .spacing(4)
    .width(260);

    container(content)
        .padding(Padding::from(16))
        .width(Length::Fixed(280.0))
        .height(Length::Fill)
        .into()
}
