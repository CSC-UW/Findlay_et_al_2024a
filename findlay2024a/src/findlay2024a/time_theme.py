from pyphi.visualize.phi_structure import DefaultTheme

_SHOW_LEGEND = False


class TimeTheme(DefaultTheme):
    DEFAULTS = dict(
        show=dict(
            cause_effect_links=False,
        ),
        layout={
            **dict(
                autosize=False,
                width=1200,
                height=1200,
                scene_camera=dict(eye=dict(x=-0.5, y=-2.0, z=0.1)),  # ???
            ),
        },
        fontfamily="Arial",
        pointsizerange=(40, 40),
        linewidthrange=(2, 5),
        linkwidthrange=(1, 1),  # ???
        geometry=dict(
            purviews=dict(
                arrange=dict(
                    max_radius=1,  # ???
                    z_offset=0.0,  # ???
                    z_spacing=0.7,  # ???
                    aspect_ratio=1,  # ???
                ),
                coordinate_kwargs=dict(
                    subset_offset_radius=0.0,
                    state_offset_radius=0.00,
                ),
            ),
            mechanisms=dict(
                arrange=dict(
                    z_offset=-0.1,
                    z_spacing=0.54,
                    radius_func="log_n_choose_k",
                    aspect_ratio=1,  # ???
                ),
            ),
        ),
        mechanisms=dict(
            showlegend=_SHOW_LEGEND,
        ),
        purviews=dict(
            hoverinfo="skip",
            showlegend=_SHOW_LEGEND,
            marker=dict(
                opacity=0.5,
                color="direction",
                colorscale="viridis",
                cmax=0.16,  # ???
                showscale=False,  # ???
            ),
        ),
        mechanism_purview_links=dict(
            showlegend=_SHOW_LEGEND,
            opacity=1,
            line=dict(
                color="rgb(120, 80, 0)",
            ),
        ),
        two_faces=dict(
            detail_threshold=10000,
            opacity=1,
            hoverinfo="skip",
            showlegend=_SHOW_LEGEND,
            line=dict(
                width="phi",
                color="rgb(242, 201, 76)",
                colorscale="viridis",
                cmax=0.02,  # ???
                showscale=False,  # ???
            ),
        ),
        three_faces=dict(
            hoverinfo="skip",  # ???
            colorscale=[[0, "rgb(102, 240, 255)"], [1.0, "rgb(49, 60, 212)"]],  # ???
            opacity=0.025,
            showscale=False,
            showlegend=_SHOW_LEGEND,
        ),
    )
