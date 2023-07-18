import plotly.express as px
from plotly import graph_objs as go


def plot_doubly_robust_band(data):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index.to_list() + data.index.to_list()[::-1],
            y=data.upper.to_list() + data.lower.to_list()[::-1],
            fill="toself",
            line={"width": 0.5, "color": "steelblue"},
        ),
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data.estimate, line_color="goldenrod", line_width=4),
    )

    fig.add_annotation(x=8, y=0.3, text=r"$Nm/Kg$", showarrow=False, font_size=22)

    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        font={"size": 25},
        xaxis={
            "tickmode": "array",
            "tickvals": [0, 100, 200],
            "ticktext": [0, 0.5, 1],
            "tickfont": {"size": 25},
        },
        template=None,
        showlegend=False,
    )

    return fig


def plot_data_presentation(data, indicator):
    data = data.query("variable == 'x'")
    data = data.reset_index()

    indicator["strike_indicator"] = indicator["strike_indicator"].map(
        {0: "Heel", 1: "Forefoot"},
    )

    data = data.merge(indicator)

    fig = px.line(
        data,
        x="time",
        y="moment",
        line_group="id",
        color="strike_indicator",
        template="simple_white",
        color_discrete_map={
            "Heel": "goldenrod",
            "Forefoot": "steelblue",
        },
    )

    fig = fig.update_traces(opacity=0.5)

    fig.add_annotation(x=8, y=0.4, text=r"$Nm/Kg$", showarrow=False, font_size=22)

    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        font={"size": 22},
        xaxis={
            "tickmode": "array",
            "tickvals": [0, 100, 200],
            "ticktext": [0, 0.5, 1],
            "tickfont": {"size": 25},
        },
        legend={
            "yanchor": "top",
            "y": 0.9,
            "xanchor": "left",
            "x": 0.45,
            "title": "",
            "font_size": 25,
        },
    )
    return fig


def plot_functional_sample(
    data,
    color_discrete_sequence=None,
    *,
    y="moment",
    opacity=0.2,
):
    data = data.reset_index()
    if not {"time", "id"} <= set(data.columns):
        msg = "'time' and 'id' needs to be either a column or an index of df."
        raise ValueError(msg)
    fig = px.line(
        data,
        x="time",
        y=y,
        color="id",
        template="simple_white",
        color_discrete_sequence=color_discrete_sequence,
    )
    update_traces_kwargs = {"opacity": opacity}
    if color_discrete_sequence is None:
        update_traces_kwargs["line_color"] = "black"
    fig = fig.update_traces(**update_traces_kwargs)
    return fig.update_layout(showlegend=False)


def plot_df_with_time_axis(data):
    if "time" not in data:
        if "time" in data.index.names:
            data = data.reset_index()
        else:
            msg = "df does not contain a time index."
            raise ValueError(msg)

    return px.line(data, x="time", y=data.columns, template="simple_white")
