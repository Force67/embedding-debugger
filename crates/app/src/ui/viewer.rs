use iced::widget::{column, container, row, shader, text, text_input, Space};
use iced::{alignment, Color, Element, Length};

use embedding_core::EmbeddingSet;
use embedding_viz::point_cloud::{PointCloud, PointCloudProgram, ViewerEvent};
use embedding_viz::ArcballCamera;

use super::app::{EmbeddingDiagnostics, Message, NeighborMatch, SelectedPointInfo};

pub fn view<'a>(
    cloud: &'a PointCloud,
    embeddings: &'a Option<EmbeddingSet>,
    selected: &'a Option<SelectedPointInfo>,
    viewer_camera: &'a ArcballCamera,
    search_query: &'a str,
    diagnostics: &'a Option<EmbeddingDiagnostics>,
    neighbors: &'a [NeighborMatch],
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

    let axis_panel = build_axis_indicator(viewer_camera);
    let selection_panel = build_selection_info(selected);
    let neighbors_panel = build_neighbors_panel(selected, neighbors);
    let diagnostics_panel = build_diagnostics_panel(diagnostics);
    let controls_panel = build_controls_hint();
    let search_panel = build_search_bar(search_query);

    let right_panel = container(
        column![
            search_panel,
            Space::with_height(8),
            axis_panel,
            Space::with_height(10),
            selection_panel,
            Space::with_height(10),
            neighbors_panel,
            Space::with_height(10),
            diagnostics_panel,
            Space::with_height(Length::Fill),
            controls_panel,
        ]
        .spacing(2),
    )
    .width(Length::Fixed(290.0))
    .padding(8);

    container(column![
        row![viewer_content, right_panel].height(Length::Fill),
        info_bar,
    ])
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}

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
    if vz.abs() > 0.72 {
        if vz > 0.0 {
            "toward"
        } else {
            "away"
        }
    } else {
        let angle = vy.atan2(vx);
        let arrows = [
            "right",
            "up-right",
            "up",
            "up-left",
            "left",
            "down-left",
            "down",
            "down-right",
        ];
        let idx = (angle / std::f32::consts::FRAC_PI_4).round() as i32;
        arrows[idx.rem_euclid(8) as usize]
    }
}

fn build_selection_info<'a>(selected: &'a Option<SelectedPointInfo>) -> Element<'a, Message> {
    match selected {
        Some(info) => column![
            text("Selected").size(11),
            text(&info.label)
                .size(14)
                .style(Color::from_rgb(1.0, 0.9, 0.5)),
            text(format!(
                "({:.3}, {:.3}, {:.3})",
                info.position[0], info.position[1], info.position[2]
            ))
            .size(10),
            text(format!("Index: {}", info.index)).size(10),
        ]
        .spacing(2)
        .into(),
        None => text("Click a point to inspect").size(11).into(),
    }
}

fn build_neighbors_panel<'a>(
    selected: &'a Option<SelectedPointInfo>,
    neighbors: &'a [NeighborMatch],
) -> Element<'a, Message> {
    if selected.is_none() {
        return text("Nearest neighbors appear after selecting a point.")
            .size(10)
            .into();
    }

    if neighbors.is_empty() {
        return column![
            text("Nearest Neighbors").size(11),
            text("No neighbors available").size(10)
        ]
        .spacing(2)
        .into();
    }

    let mut col = column![text("Nearest Neighbors").size(11)].spacing(2);
    for n in neighbors.iter().take(8) {
        col = col.push(
            text(format!(
                "#{} {}  cos {:.4}  l2 {:.3}",
                n.index, n.label, n.cosine, n.l2
            ))
            .size(10),
        );
    }
    col.into()
}

fn build_diagnostics_panel<'a>(
    diagnostics: &'a Option<EmbeddingDiagnostics>,
) -> Element<'a, Message> {
    let Some(d) = diagnostics else {
        return text("Diagnostics appear after embedding generation.")
            .size(10)
            .into();
    };

    let mut col = column![
        text("Diagnostics").size(11),
        text(format!("Vectors: {} × {}", d.count, d.dimensions)).size(10),
        text(format!(
            "Norms min/mean/max: {:.3} / {:.3} / {:.3}",
            d.min_norm, d.mean_norm, d.max_norm
        ))
        .size(10),
        text(format!("Norm stddev: {:.3}", d.std_norm)).size(10),
        text(format!(
            "Finite: {}  Non-finite: {}  Zero: {}",
            d.finite_vectors, d.non_finite_vectors, d.zero_vectors
        ))
        .size(10),
        text(format!(
            "Exact dupes: {}  Near dupes: {} (sampled {})",
            d.exact_duplicates, d.near_duplicates, d.near_duplicate_pairs_sampled
        ))
        .size(10),
    ]
    .spacing(2);

    if !d.near_examples.is_empty() {
        col = col.push(text("Near-duplicate examples").size(10));
        for ex in d.near_examples.iter().take(4) {
            col = col.push(text(format!("{} ~ {} ({:.4})", ex.left, ex.right, ex.cosine)).size(10));
        }
    }

    col.into()
}

fn build_controls_hint<'a>() -> Element<'a, Message> {
    column![
        text("Controls").size(11),
        text("Arrow keys Rotate").size(10),
        text("WASD   Pan").size(10),
        text("Q/E    Zoom").size(10),
        text("Scroll Zoom").size(10),
        text("R      Reset").size(10),
        text("F      Focus").size(10),
        text("Esc    Deselect").size(10),
        text("Dbl-click Reset").size(10),
        text("Adaptive LOD on drag").size(10),
    ]
    .spacing(1)
    .into()
}

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
