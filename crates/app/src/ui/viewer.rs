use iced::widget::{column, container, shader, text, Space};
use iced::{alignment, Element, Length};

use embedding_core::EmbeddingSet;
use embedding_viz::point_cloud::{PointCloud, PointCloudProgram};

use super::app::Message;

pub fn view<'a>(cloud: &'a PointCloud, embeddings: &'a Option<EmbeddingSet>) -> Element<'a, Message> {
    let has_data = !cloud.points.is_empty();

    let viewer_content: Element<'a, Message> = if has_data {
        // Render the 3D point cloud via iced's shader widget.
        let program = PointCloudProgram {
            cloud: cloud.clone(),
        };
        let shader_elem: Element<'a, ()> = shader(program)
            .width(Length::Fill)
            .height(Length::Fill)
            .into();
        shader_elem.map(|_| Message::Noop)
    } else {
        // Empty state placeholder.
        container(
            column![
                Space::with_height(Length::Fill),
                text("No embeddings to display").size(20),
                text("Configure a provider, load tokens, and generate embeddings.").size(14),
                Space::with_height(Length::Fill),
            ]
            .align_items(alignment::Alignment::Center)
            .spacing(8),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x()
        .center_y()
        .into()
    };

    let info_bar = match embeddings {
        Some(set) => text(format!(
            "{} · {}D · {} points",
            set.model,
            set.dimensions,
            set.len()
        ))
        .size(12),
        None => text("").size(12),
    };

    container(
        column![
            viewer_content,
            info_bar,
        ]
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}
