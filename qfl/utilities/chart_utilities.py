import matplotlib.pyplot as plt


def format_plot(ax=None, text_color_scheme=None, background_color=None):
    ax.title.set_color(text_color_scheme)
    ax.spines['bottom'].set_color(text_color_scheme)
    ax.spines['top'].set_color(text_color_scheme)
    ax.xaxis.label.set_color(text_color_scheme)
    ax.yaxis.label.set_color(text_color_scheme)
    ax.tick_params(axis='x', colors=text_color_scheme)
    ax.tick_params(axis='y', colors=text_color_scheme)
    ax.set_axis_bgcolor(background_color)