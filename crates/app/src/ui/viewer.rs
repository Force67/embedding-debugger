use iced::widget::{column, container, row, shader, text, text_input, Space};
use iced::{alignment, Color, Element, Length};

use embedding_core::EmbeddingSet;
use embedding_viz::point_cloud::{PointCloud, PointCloudProgram, ViewerEvent};
use embedding_viz::ArcballCamera;

use super::app::{Message, SelectedPointInfo};

pub fn view<'a>(
    cloud: &'a PointCloud,
    embeddings: &'a Option<EmbeddingSet>,
    selected: &'a Option<SelectedPointInfo>,
    viewer_camera: &'a ArcballCamera,
    search_query: &'a str,
) -> Element<'a, Message> {
    let has_data = !cloud.points.is_empty();

    let viewer_content: Element<'a, Message> = if has_data {
        let program = PointCloudProgram {
            cloud: cloud.clone(),
        };
        let shader_elem: Element<'a, ViewerEvent> = shader(program)
            .width(Length::Fill)
            .height(Length::Fill)
            .into();
        shader_elem.map(|evt| match evt {
            ViewerEvent::PointSelected(sel) => Message::PointSelected(sel),
            ViewerEvent::CameraChanged(cam) => Message::ViewerCameraChanged(cam),
        })
    } else {
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

    // ── Right-hand info panel ────────────────────────────────────────────
    let axis_panel = build_axis_indicator(viewer_camera);
    let selection_panel = build_selection_info(selected);
    let controls_panel = build_controls_hint();
    let search_panel = build_search_bar(search_query);

    let right_panel = container(
        column![
            search_panel,
            Space::with_height(8),
            axis_panel,
            Space::with_height(12),
            selection_panel,
            Space::with_height(Length::Fill),
            controls_panel,
        ]
        .spacing(2),
    )
    .width(Length::Fixed(155.0))
    .padding(8);

    container(
        column![
            row![
                viewer_content,
                right_panel,
            ]
            .height(Length::Fill),
            info_bar,
        ]
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}

// ─── Axis direction indicator ────────────────────────────────────────────────

fn build_axis_indicator<'a>(camera: &ArcballCamera) -> Element<'a, Message> {
    let x_view = camera.project_axis([1.0, 0.0, 0.0]);
    let y_view = camera.project_axis([0.0, 1.0, 0.0]);
    let z_view = camera.project_axis([0.0, 0.0, 1.0]);

    column![
        text("Orientation").size(11),
        row![
            text("X ").size(13).style(Color::from_rgb(1.0, 0.4, 0.35)),
            text(axis_arrow(x_view)).size(13),
        ]
        .spacing(4),
        row![
            text("Y ").size(13).style(Color::from_rgb(0.35, 1.0, 0.45)),
            text(axis_arrow(y_view)).size(13),
        ]
        .spacing(4),
        row![
            text("Z ").size(13).style(Color::from_rgb(0.35, 0.55, 1.0)),
            text(axis_arrow(z_view)).size(13),
        ]
        .spacing(4),
    ]
    .spacing(2)
    .into()
}

fn axis_arrow(view_dir: [f32; 3]) -> &'static str {
    let (vx, vy, vz) = (view_dir[0], view_dir[1], view_dir[2]);
    // If primarily going into/out of the screen.
    if vz.abs() > 0.72 {
        if vz > 0.0 { "⊙ toward" } else { "⊗ away" }
    } else {
        let angle = vy.atan2(vx);
        let arrows = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"];
        let idx = (angle / std::f32::consts::FRAC_PI_4).round() as i32;
        arrows[idx.rem_euclid(8) as usize]
    }
}

// ─── Selected point info ─────────────────────────────────────────────────────

fn build_selection_info<'a>(selected: &'a Option<SelectedPointInfo>) -> Element<'a, Message> {
    match selected {
        Some(info) => {
            column![
                text("Selected").size(11),
                text(&info.label).size(14).style(Color::from_rgb(1.0, 0.9, 0.5)),
                text(format!(
                    "({:.3}, {:.3}, {:.3})",
                    info.position[0], info.position[1], info.position[2]
                ))
                .size(10),
                text(format!("Index: {}", info.index)).size(10),
            ]
            .spacing(2)
            .into()
        }
        None => {
            text("Click a point to inspect").size(11).into()
        }
    }
}

// ─── Controls hint ───────────────────────────────────────────────────────────

fn build_controls_hint<'a>() -> Element<'a, Message> {
    column![
        text("Controls").size(11),
        text("↑↓←→  Rotate").size(10),
        text("WASD   Pan").size(10),
        text("Q/E    Zoom").size(10),
        text("Scroll Zoom").size(10),
        text("R      Reset").size(10),
        text("F      Focus").size(10),
        text("Esc    Deselect").size(10),
        text("Dbl-click Reset").size(10),
    ]
    .spacing(1)
    .into()
}

// ─── Search bar ──────────────────────────────────────────────────────────────

fn build_search_bar<'a>(query: &'a str) -> Element<'a, Message> {
    column![
        text("Search").size(11),
        text_input("find a word…", query)
            .on_input(Message::SearchQueryChanged)
            .size(12)
            .width(Length::Fill),
    ]
    .spacing(3)
    .into()
}
