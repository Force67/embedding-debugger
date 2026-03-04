use iced::widget::{button, column, container, pick_list, row, slider, text, text_input, Space};
use iced::{Element, Length, Padding};

use embedding_core::TokenCollection;

use super::app::Message;

/// State for the sidebar UI controls.
#[derive(Debug, Clone)]
pub struct SidebarState {
    pub selected_profile: Option<String>,
    pub profile_name_input: String,
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
    /// Currently selected projection method label ("PCA" or "t-SNE").
    pub projection_input: String,
    /// Current t-SNE perplexity value.
    pub perplexity_input: f32,
}

impl Default for SidebarState {
    fn default() -> Self {
        Self {
            selected_profile: None,
            profile_name_input: String::new(),
            provider_input: String::new(),
            model_input: String::new(),
            api_key_input: String::new(),
            dimensions_input: String::new(),
            endpoint_input: String::new(),
            deployment_input: String::new(),
            tokens_text: String::new(),
            projection_input: "PCA".to_string(),
            perplexity_input: 30.0,
        }
    }
}

const PROVIDERS: &[&str] = &["OpenRouter", "OpenAI", "Azure"];
const PROJECTIONS: &[&str] = &["PCA", "t-SNE"];

pub fn view<'a>(
    state: &'a SidebarState,
    profiles: Vec<String>,
    tokens: &'a Option<TokenCollection>,
    loading: bool,
    bubble_size: f32,
) -> Element<'a, Message> {
    let is_azure = state.provider_input == "Azure";

    let profile_picker = pick_list(
        profiles,
        state.selected_profile.clone(),
        Message::ProfileSelected,
    )
    .width(Length::Fill);

    let profile_name_input = text_input("Profile name", &state.profile_name_input)
        .on_input(Message::ProfileNameChanged)
        .width(Length::Fill);

    let save_profile_button = if state.profile_name_input.trim().is_empty() {
        button(text("Save profile")).width(Length::FillPortion(1))
    } else {
        button(text("Save profile"))
            .on_press(Message::SaveProfilePressed)
            .width(Length::FillPortion(1))
    };

    let delete_profile_button = if state.selected_profile.is_none() {
        button(text("Delete")).width(Length::FillPortion(1))
    } else {
        button(text("Delete"))
            .on_press(Message::DeleteProfilePressed)
            .width(Length::FillPortion(1))
    };

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
        Some(collection) => text(format!("{}: {} tokens", collection.name, collection.len())),
        None => text("No tokens loaded"),
    };

    let projection_picker = pick_list(
        PROJECTIONS.to_vec(),
        Some(if state.projection_input.is_empty() {
            "PCA"
        } else {
            state.projection_input.as_str()
        }),
        |s| Message::ProjectionMethodChanged(s.to_string()),
    )
    .width(Length::Fill);

    let is_tsne = state.projection_input == "t-SNE";

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
        let model_input = text_input(
            "Model (e.g. openai/text-embedding-3-small)",
            &state.model_input,
        )
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

    let mut content = column![
        text("Profiles").size(14),
        profile_picker,
        profile_name_input,
        row![save_profile_button, delete_profile_button,].spacing(6),
        Space::with_height(12),
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
        Space::with_height(16),
        text("Projection").size(14),
        projection_picker,
    ]
    .spacing(4)
    .width(260);

    if is_tsne {
        content = content.push(Space::with_height(4));
        content = content.push(text(format!("Perplexity  {:.0}", state.perplexity_input)).size(14));
        content = content.push(
            slider(
                5.0_f32..=50.0,
                state.perplexity_input,
                Message::PerplexityChanged,
            )
            .step(1.0_f32)
            .width(Length::Fill),
        );
    }

    content = content.push(Space::with_height(16));
    content = content.push(text(format!("Bubble Size  {:.0}", bubble_size)).size(14));
    content = content.push(
        slider(2.0_f32..=30.0, bubble_size, Message::BubbleSizeChanged)
            .step(0.5_f32)
            .width(Length::Fill),
    );

    container(content)
        .padding(Padding::from(16))
        .width(Length::Fixed(280.0))
        .height(Length::Fill)
        .into()
}
