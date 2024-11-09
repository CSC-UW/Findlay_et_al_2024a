from pyphi.visualize.phi_structure import DefaultTheme


# def _italicize(text):
#    return r"<i>" + "".join(text) + r"</i>"


class PubTheme(DefaultTheme):
    DEFAULTS = dict(
        show=dict(
            cause_effect_links=False,
        ),
        labels=dict(
            # postprocessor=_italicize,
            postprocessor=None,
        ),
        layout={
            **dict(
                autosize=False,
                showlegend=False,
                width=1800,
                height=1400,
                scene_camera=dict(eye=dict(x=2.5, y=1.5, z=0.15)),  # Undocumented
            ),
        },
        fontfamily="Arial",
        fontsize=30,
        pointsizerange=(50, 50),
        linewidthrange=(2, 5),
        linkwidthrange=(2, 5),  # Undocumented
        geometry=dict(
            purviews=dict(
                arrange=dict(
                    max_radius=1,  # Undocumented
                    z_offset=0.0,  # Undocumented
                    z_spacing=1,  # Undocumented
                    aspect_ratio=1,  # Undocumented
                ),
                coordinate_kwargs=dict(
                    direction_offset=0.3,
                    subset_offset_radius=0.0,
                    state_offset_radius=0.00,
                ),
            ),
            mechanisms=dict(
                arrange=dict(
                    max_radius=0.5,
                    z_offset=-0.250,
                    z_spacing=1.75,
                    radius_func="log_n_choose_k",
                    aspect_ratio=1,
                ),
            ),
        ),
        mechanisms=dict(
            showlegend=False,
        ),
        purviews=dict(
            hoverinfo="skip",
            showlegend=False,
            marker=dict(
                opacity=0.5,
                colorscale="viridis",
                cmin=0,
                cmax=1,  # Undocumented
            ),
        ),
        mechanism_purview_links=dict(
            showlegend=False,
            opacity=1,
            line=dict(
                color="orange",
                width="phi",
            ),
        ),
        two_faces=dict(
            hoverinfo="skip",
            showlegend=False,
            line=dict(
                width="phi",
                color="phi",
                colorscale="blues",  # Was supposed to be "reds", but was ignored.
                cmin=0,
                cmax=1,  # Undocumented
            ),
        ),
        three_faces=dict(
            hoverinfo="skip",  # Undocumented
            colorscale="blues",  # Was supposed to be "viridis", but was ignored.
            opacity=0.1,  # Was supposed to be 0.025, but was ignored.
            cmin=0,
            cmax=1,  # Undocumented
            showlegend=False,
        ),
    )
