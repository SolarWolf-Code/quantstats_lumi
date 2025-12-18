#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Quantreturns: Portfolio analytics for quants
# https://github.com/ranaroussi/quantreturns
#
# Copyright 2019-2023 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as _plt

try:
    _plt.rcParams["font.family"] = "Arial"
except Exception:
    pass

import matplotlib.dates as _mdates
import numpy as _np
import pandas as _pd
import seaborn as _sns
from matplotlib.ticker import FormatStrFormatter as _FormatStrFormatter
from matplotlib.ticker import FuncFormatter as _FuncFormatter

from .. import stats as _stats
from .. import utils as _utils

_sns.set(
    font_scale=1.1,
    rc={
        "figure.figsize": (10, 6),
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": "#dddddd",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "text.color": "#333333",
        "xtick.color": "#666666",
        "ytick.color": "#666666",
    },
)

def _hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    """Convert RGB tuple to hex color"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _get_luminance(hex_color):
    """Calculate relative luminance of a color (0-1 scale)"""
    rgb = _hex_to_rgb(hex_color)
    # Normalize RGB values to 0-1
    rgb_norm = [val / 255.0 for val in rgb]
    # Apply gamma correction
    rgb_linear = [
        val / 12.92 if val <= 0.03928 else ((val + 0.055) / 1.055) ** 2.4
        for val in rgb_norm
    ]
    # Calculate luminance using relative luminance formula
    luminance = 0.2126 * rgb_linear[0] + 0.7152 * rgb_linear[1] + 0.0722 * rgb_linear[2]
    return luminance


def _is_dark_color(hex_color):
    """Determine if a color is dark (luminance < 0.5)"""
    return _get_luminance(hex_color) < 0.5


def _darken_color(hex_color, factor=0.15):
    """
    Darken a hex color by a given factor (0-1).
    
    Parameters
    ----------
    hex_color : str
        Hex color code (e.g., '#111b29')
    factor : float
        Darkening factor (0-1), default is 0.15 (15% darker)
    
    Returns
    -------
    str
        Darkened hex color code
    """
    if hex_color is None or hex_color == 'white':
        return '#f0f0f0'  # Light gray for white backgrounds
    
    # Normalize hex color
    if not hex_color.startswith('#'):
        hex_color = '#' + hex_color
    hex_color = hex_color.lower()
    
    # Convert to RGB
    rgb = _hex_to_rgb(hex_color)
    
    # Darken each component
    darkened_rgb = tuple(max(0, int(val * (1 - factor))) for val in rgb)
    
    # Convert back to hex
    return _rgb_to_hex(darkened_rgb)


def _derive_theme_colors(background_theme=None, font_theme='dark'):
    """
    Derive a color scheme from background theme and font theme.
    
    Parameters
    ----------
    background_theme : str or None
        Hex color code (e.g., '#111b29') or None for default white background
    font_theme : str
        Either 'light' or 'dark' to specify font color theme
    
    Returns
    -------
    dict
        Dictionary with bg_color, text_color, subtitle_color, grid_color, etc.
    """
    # Default background
    if background_theme is None:
        bg_color = 'white'
        metrics_bg_color = '#f5f5f5'  # Light gray for metrics area
    else:
        # Normalize hex color
        if not background_theme.startswith('#'):
            background_theme = '#' + background_theme
        bg_color = background_theme.lower()
        # Create a darker shade for metrics area (tables, parameters)
        metrics_bg_color = _darken_color(bg_color, factor=0.15)
    
    # Set font colors based on font_theme
    if font_theme.lower() == 'light':
        # Light text (for dark backgrounds)
        text_color = '#ffffff'
        subtitle_color = '#cccccc'
        grid_color = '#666666'
        edge_color = '#ffffff'
        legend_bg = bg_color
        legend_edge = '#ffffff'
        # For metrics area with darker background, use slightly dimmer text
        metrics_text_color = '#e0e0e0'
        metrics_header_bg = metrics_bg_color
    else:  # 'dark' or default
        # Dark text (for light backgrounds)
        text_color = '#000000'
        subtitle_color = '#666666'
        grid_color = '#dddddd'
        edge_color = '#333333'
        legend_bg = bg_color
        legend_edge = '#cccccc'
        # For metrics area, use same text color
        metrics_text_color = text_color
        metrics_header_bg = metrics_bg_color
    
    return {
        'bg_color': bg_color,
        'text_color': text_color,
        'subtitle_color': subtitle_color,
        'grid_color': grid_color,
        'edge_color': edge_color,
        'legend_bg': legend_bg,
        'legend_edge': legend_edge,
        'metrics_bg_color': metrics_bg_color,
        'metrics_text_color': metrics_text_color,
        'metrics_header_bg': metrics_header_bg,
    }


# Context manager for theme
class _ThemeContext:
    """Context manager to temporarily apply custom theme to matplotlib"""
    def __init__(self, background_theme=None, font_theme='dark'):
        self.background_theme = background_theme
        self.font_theme = font_theme
        self.original_params = {}
        self.theme_colors = _derive_theme_colors(background_theme, font_theme)
    
    def __enter__(self):
        if self.background_theme is not None:
            # Save original parameters
            self.original_params = {
                'figure.facecolor': _plt.rcParams.get('figure.facecolor'),
                'axes.facecolor': _plt.rcParams.get('axes.facecolor'),
                'axes.edgecolor': _plt.rcParams.get('axes.edgecolor'),
                'axes.labelcolor': _plt.rcParams.get('axes.labelcolor'),
                'text.color': _plt.rcParams.get('text.color'),
                'xtick.color': _plt.rcParams.get('xtick.color'),
                'ytick.color': _plt.rcParams.get('ytick.color'),
                'axes.grid': _plt.rcParams.get('axes.grid'),
                'grid.color': _plt.rcParams.get('grid.color'),
                'grid.linewidth': _plt.rcParams.get('grid.linewidth'),
            }

            # Apply theme colors
            _plt.rcParams.update({
                'figure.facecolor': self.theme_colors['bg_color'],
                'axes.facecolor': self.theme_colors['bg_color'],
                'axes.edgecolor': self.theme_colors['edge_color'],
                'axes.labelcolor': self.theme_colors['text_color'],
                'text.color': self.theme_colors['text_color'],
                'xtick.color': self.theme_colors['text_color'],
                'ytick.color': self.theme_colors['text_color'],
                'axes.grid': True,
                'grid.color': self.theme_colors['grid_color'],
                'grid.linewidth': 0.5,
            })
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.background_theme is not None and self.original_params:
            _plt.rcParams.update(self.original_params)

_FLATUI_COLORS = [
    "#FEDD78",
    "#348DC1",
    "#BA516B",
    "#4FA487",
    "#9B59B6",
    "#613F66",
    "#84B082",
    "#DC136C",
    "#559CAD",
    "#4A5899",
]
_GRAYSCALE_COLORS = [
    "#000000",
    "#222222",
    "#555555",
    "#888888",
    "#AAAAAA",
    "#CCCCCC",
    "#EEEEEE",
    "#333333",
    "#666666",
    "#999999",
]


def _get_colors(grayscale):
    colors = _FLATUI_COLORS
    ls = "-"
    alpha = 0.8
    if grayscale:
        colors = _GRAYSCALE_COLORS
        ls = "-"
        alpha = 0.5
    return colors, ls, alpha




def plot_returns_bars(
        returns,
        benchmark=None,
        returns_label="Strategy",
        hline=None,
        hlw=None,
        hlcolor="red",
        hllabel="",
        resample="YE",
        title="Returns",
        match_volatility=False,
        log_scale=False,
        figsize=(10, 6),
        grayscale=False,
        background_theme=None,
        font_theme='dark',
        fontname="Arial",
        ylabel=True,
        subtitle=True,
        savefig=None,
        show=True,
):
    # Grayscale and background_theme are mutually exclusive
    if grayscale:
        background_theme = None
    if match_volatility and benchmark is None:
        raise ValueError("match_volatility requires passing of " "benchmark.")
    if match_volatility and benchmark is not None:
        bmark_vol = benchmark.loc[returns.index].std()
        returns = (returns / returns.std()) * bmark_vol

    # ---------------
    colors, _, _ = _get_colors(grayscale)
    if isinstance(returns, _pd.Series):
        df = _pd.DataFrame(index=returns.index, data={returns.name: returns})
    elif isinstance(returns, _pd.DataFrame):
        df = _pd.DataFrame(
            index=returns.index, data={col: returns[col] for col in returns.columns}
        )
    if isinstance(benchmark, _pd.Series):
        df[benchmark.name] = benchmark[benchmark.index.isin(returns.index)]
        if isinstance(returns, _pd.Series):
            df = df[[benchmark.name, returns.name]]
        elif isinstance(returns, _pd.DataFrame):
            col_names = [benchmark.name, returns.columns]
            df = df[list(_pd.core.common.flatten(col_names))]

    df = df.dropna()
    if resample is not None:
        df = df.resample(resample).apply(_stats.comp).resample(resample).last()
    # ---------------

    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Derive theme colors
    theme_colors = _derive_theme_colors(background_theme, font_theme)
    text_color = theme_colors['text_color']
    subtitle_color = theme_colors['subtitle_color']
    bg_color = theme_colors['bg_color']
    grid_color = theme_colors['grid_color']
    edge_color = theme_colors['edge_color']
    legend_bg = theme_colors['legend_bg']
    legend_edge = theme_colors['legend_edge']
    
    with _ThemeContext(background_theme, font_theme):
        fig.suptitle(
            title, y=0.94, fontweight="bold", fontname=fontname, fontsize=14, color=text_color
        )

        if subtitle:
            ax.set_title(
                "%s - %s           \n"
                % (
                    df.index.date[:1][0].strftime("%Y"),
                    df.index.date[-1:][0].strftime("%Y"),
                ),
                fontsize=12,
                color=subtitle_color,
            )

        if benchmark is None:
            colors = colors[1:]
        df.plot(kind="bar", ax=ax, color=colors)

        fig.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        if background_theme is not None:
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            for spine in ax.spines.values():
                spine.set_color(edge_color)
            ax.grid(True, color=grid_color, linewidth=0.5, alpha=1.0)

        try:
            ax.set_xticklabels(df.index.year)
            years = sorted(list(set(df.index.year)))
        except AttributeError:
            ax.set_xticklabels(df.index)
            years = sorted(list(set(df.index)))

        # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')
        # years = sorted(list(set(df.index.year)))
        if len(years) > 10:
            mod = int(len(years) / 10)
            _plt.xticks(
                _np.arange(len(years)),
                [str(year) if not i % mod else "" for i, year in enumerate(years)],
            )

        # rotate and align the tick labels so they look better
        fig.autofmt_xdate()

        if hline is not None:
            if not isinstance(hline, _pd.Series):
                if grayscale:
                    hlcolor = "gray" if background_theme is None else text_color
                ax.axhline(hline, ls="--", lw=hlw, color=hlcolor, label=hllabel, zorder=2)

        ax.axhline(0, ls="--", lw=1, color=text_color if background_theme is not None else "#000000", zorder=2)

        # if isinstance(benchmark, _pd.Series) or hline:
        legend = ax.legend(fontsize=11)
        if background_theme is not None and legend is not None:
            for text in legend.get_texts():
                text.set_color(text_color)
            # Style the legend frame for theme
            legend.get_frame().set_facecolor(legend_bg)
            legend.get_frame().set_edgecolor(legend_edge)
            legend.get_frame().set_alpha(0.8)

        _plt.yscale("symlog" if log_scale else "linear")

        ax.set_xlabel("")
        if ylabel:
            ax.set_ylabel(
                "Returns", fontname=fontname, fontweight="bold", fontsize=12, color=text_color
            )
            ax.yaxis.set_label_coords(-0.1, 0.5)

        if benchmark is None and len(_pd.DataFrame(returns).columns) == 1:
            ax.get_legend().remove()

        try:
            _plt.subplots_adjust(hspace=0, bottom=0, top=1)
        except Exception:
            pass

        try:
            fig.tight_layout()
        except Exception:
            pass

        if savefig:
            if isinstance(savefig, dict):
                _plt.savefig(**savefig)
            else:
                _plt.savefig(savefig)

        if show:
            _plt.show(block=False)

        _plt.close()

        if not show:
            return fig

        return None


def plot_timeseries(
        returns,
        benchmark=None,
        title="Returns",
        compound=False,
        cumulative=True,
        fill=False,
        returns_label="Strategy",
        hline=None,
        hlw=None,
        hlcolor="red",
        hllabel="",
        percent=True,
        match_volatility=False,
        log_scale=False,
        resample=None,
        lw=1.5,
        figsize=(10, 6),
        ylabel="",
        grayscale=False,
        background_theme=None,
        font_theme='dark',
        fontname="Arial",
        subtitle=True,
        savefig=None,
        show=True,
):
    # Grayscale and background_theme are mutually exclusive
    if grayscale:
        background_theme = None
    
    # Derive theme colors
    theme_colors = _derive_theme_colors(background_theme, font_theme)
    text_color = theme_colors['text_color']
    subtitle_color = theme_colors['subtitle_color']
    bg_color = theme_colors['bg_color']
    grid_color = theme_colors['grid_color']
    edge_color = theme_colors['edge_color']
    legend_bg = theme_colors['legend_bg']
    legend_edge = theme_colors['legend_edge']
    
    with _ThemeContext(background_theme, font_theme):
        colors, ls, alpha = _get_colors(grayscale)
        
        returns = returns.fillna(0)
        if isinstance(benchmark, _pd.Series):
            benchmark = benchmark.fillna(0)

        if match_volatility and benchmark is None:
            raise ValueError("match_volatility requires passing of " "benchmark.")
        if match_volatility and benchmark is not None:
            bmark_vol = benchmark.std()
            returns = (returns / returns.std()) * bmark_vol

        # ---------------
        if compound is True:
            if cumulative:
                returns = _stats.compsum(returns)
                if isinstance(benchmark, _pd.Series):
                    benchmark = _stats.compsum(benchmark)
            else:
                returns = returns.cumsum()
                if isinstance(benchmark, _pd.Series):
                    benchmark = benchmark.cumsum()

        if resample:
            returns = returns.resample(resample)
            returns = returns.last() if compound is True else returns.sum()
            if isinstance(benchmark, _pd.Series):
                benchmark = benchmark.resample(resample)
                benchmark = benchmark.last() if compound is True else benchmark.sum()
        # ---------------

        fig, ax = _plt.subplots(figsize=figsize)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        
        fig.suptitle(
            title, y=0.94, fontweight="bold", fontname=fontname, fontsize=14, color=text_color
        )

        if subtitle:
            ax.set_title(
                "%s - %s            \n"
                % (
                    returns.index.date[:1][0].strftime("%e %b '%y"),
                    returns.index.date[-1:][0].strftime("%e %b '%y"),
                ),
                fontsize=12,
                color=subtitle_color,
            )

        fig.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        if isinstance(benchmark, _pd.Series):
            ax.plot(benchmark, lw=lw, ls=ls, label=benchmark.name, color=colors[0])

        alpha = 0.25 if grayscale else 1
        if isinstance(returns, _pd.Series):
            ax.plot(returns, lw=lw, label=returns.name, color=colors[1], alpha=alpha)
        elif isinstance(returns, _pd.DataFrame):
            # color_dict = {col: colors[i+1] for i, col in enumerate(returns.columns)}
            for i, col in enumerate(returns.columns):
                ax.plot(returns[col], lw=lw, label=col, alpha=alpha, color=colors[i + 1])

        if fill:
            if isinstance(returns, _pd.Series):
                ax.fill_between(returns.index, 0, returns, color=colors[1], alpha=0.25)
            elif isinstance(returns, _pd.DataFrame):
                for i, col in enumerate(returns.columns):
                    ax.fill_between(
                        returns[col].index, 0, returns[col], color=colors[i + 1], alpha=0.25
                    )

        # rotate and align the tick labels so they look better
        fig.autofmt_xdate()

        # use a more precise date string for the x axis locations in the toolbar
        # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')

        if hline is not None:
            if not isinstance(hline, _pd.Series):
                if grayscale:
                    hlcolor = "black" if background_theme is None else text_color
                ax.axhline(hline, ls="--", lw=hlw, color=hlcolor, label=hllabel, zorder=2)

        ax.axhline(0, ls="-", lw=1, color="gray" if background_theme is None else grid_color, zorder=1)
        ax.axhline(0, ls="--", lw=1, color=text_color if (grayscale or background_theme is not None) else "black", zorder=2)

        # if isinstance(benchmark, _pd.Series) or hline is not None:
        legend = ax.legend(fontsize=11)
        if background_theme is not None and legend is not None:
            for text in legend.get_texts():
                text.set_color(text_color)
            # Style the legend frame for theme
            legend.get_frame().set_facecolor(legend_bg)
            legend.get_frame().set_edgecolor(legend_edge)
            legend.get_frame().set_alpha(0.8)

        _plt.yscale("symlog" if log_scale else "linear")

        # Set y-axis limits to avoid blank space at the bottom and top
        min_val = returns.min()
        max_val = returns.max()
        if benchmark is not None:
            min_val = min(min_val, benchmark.min())
            max_val = max(max_val, benchmark.max())

        # Handle cases where min_val or max_val might be NaN or Inf
        if not _np.isfinite(min_val) or not _np.isfinite(max_val) or min_val == max_val:
            min_val = -1  # Default min value
            max_val = 1   # Default max value
            # if using percent, adjust defaults
            if percent:
                min_val = -0.01
                max_val = 0.01

        ax.set_ylim(bottom=min_val, top=max_val)

        if percent:
            ax.yaxis.set_major_formatter(_FuncFormatter(format_pct_axis))
            # ax.yaxis.set_major_formatter(_plt.FuncFormatter(
            #     lambda x, loc: "{:,}%".format(int(x*100))))

        ax.set_xlabel("")
        if ylabel:
            ax.set_ylabel(
                ylabel, fontname=fontname, fontweight="bold", fontsize=12, color=text_color
            )
        ax.yaxis.set_label_coords(-0.1, 0.5)
        
        # Update tick colors for theme
        if background_theme is not None:
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            for spine in ax.spines.values():
                spine.set_color(edge_color)

        if benchmark is None and len(_pd.DataFrame(returns).columns) == 1:
            ax.get_legend().remove()

        try:
            _plt.subplots_adjust(hspace=0, bottom=0, top=1)
        except Exception:
            pass

        try:
            fig.tight_layout()
        except Exception:
            pass

        if savefig:
            if isinstance(savefig, dict):
                _plt.savefig(**savefig)
            else:
                _plt.savefig(savefig)

        if show:
            _plt.show(block=False)

        _plt.close()

        if not show:
            return fig

        return None


def plot_histogram(
        returns,
        benchmark,
        resample="ME",
        bins=20,
        fontname="Arial",
        grayscale=False,
        background_theme=None,
        font_theme='dark',
        title="Returns",
        kde=True,
        figsize=(10, 6),
        ylabel=True,
        subtitle=True,
        compounded=True,
        savefig=None,
        show=True,
):
    # Grayscale and background_theme are mutually exclusive
    if grayscale:
        background_theme = None
    
    # Derive theme colors
    theme_colors = _derive_theme_colors(background_theme, font_theme)
    text_color = theme_colors['text_color']
    subtitle_color = theme_colors['subtitle_color']
    bg_color = theme_colors['bg_color']
    grid_color = theme_colors['grid_color']
    edge_color = theme_colors['edge_color']
    legend_bg = theme_colors['legend_bg']
    legend_edge = theme_colors['legend_edge']
    # colors = ['#348dc1', '#003366', 'red']
    # if grayscale:
    #     colors = ['silver', 'gray', 'black']

    colors, _, _ = _get_colors(grayscale)

    apply_fnc = _stats.comp if compounded else 'sum'
    if benchmark is not None:
        benchmark = (
            benchmark.fillna(0)
            .resample(resample)
            .apply(apply_fnc)
            .resample(resample)
            .last()
        )

    returns = (
        returns.fillna(0).resample(resample).apply(apply_fnc).resample(resample).last()
    )

    figsize = (0.995 * figsize[0], figsize[1])
    fig, ax = _plt.subplots(figsize=figsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    
    with _ThemeContext(background_theme, font_theme):
        fig.suptitle(
            title, y=0.94, fontweight="bold", fontname=fontname, fontsize=14, color=text_color
        )

        if subtitle:
            ax.set_title(
                "%s - %s           \n"
                % (
                    returns.index.date[:1][0].strftime("%Y"),
                    returns.index.date[-1:][0].strftime("%Y"),
                ),
                fontsize=12,
                color=subtitle_color,
            )

        fig.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        if background_theme is not None:
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            for spine in ax.spines.values():
                spine.set_color(edge_color)
            ax.grid(True, color=grid_color, linewidth=0.5, alpha=1.0)

        if isinstance(returns, _pd.DataFrame) and len(returns.columns) == 1:
            returns = returns[returns.columns[0]]

        pallete = colors[1:2] if benchmark is None else colors[:2]
        alpha = 0.7
        if isinstance(returns, _pd.DataFrame):
            pallete = (
                colors[1: len(returns.columns) + 1]
                if benchmark is None
                else colors[: len(returns.columns) + 1]
            )
            if len(returns.columns) > 1:
                alpha = 0.5

        def fix_instance(x):
            return x[x.columns[0]] if isinstance(x, _pd.DataFrame) else x
        if benchmark is not None:
            if isinstance(returns, _pd.Series):
                combined_returns = (
                    fix_instance(benchmark).to_frame()
                    .join(returns.to_frame())
                    .stack()
                    .reset_index()
                    .rename(columns={"level_1": "", 0: "Returns"})
                )
            elif isinstance(returns, _pd.DataFrame):
                combined_returns = (
                    fix_instance(benchmark).to_frame()
                    .join(returns)
                    .stack()
                    .reset_index()
                    .rename(columns={"level_1": "", 0: "Returns"})
                )
            _sns.histplot(
                data=combined_returns,
                x="Returns",
                bins=bins,
                alpha=alpha,
                kde=kde,
                stat="density",
                hue="",
                palette=pallete,
                ax=ax,
            )

        else:
            if isinstance(returns, _pd.Series):
                combined_returns = returns.copy()
                if kde:
                    _sns.kdeplot(
                        data=combined_returns,
                        color=text_color,
                        ax=ax,
                        warn_singular=False,
                    )
                _sns.histplot(
                    data=combined_returns,
                    bins=bins,
                    alpha=alpha,
                    kde=False,
                    stat="density",
                    color=colors[1],
                    ax=ax,
                )

            elif isinstance(returns, _pd.DataFrame):
                combined_returns = (
                    returns.stack()
                    .reset_index()
                    .rename(columns={"level_1": "", 0: "Returns"})
                )
                # _sns.kdeplot(data=combined_returns, color='black', ax=ax, warn_singular=False)
                _sns.histplot(
                    data=combined_returns,
                    x="Returns",
                    bins=bins,
                    alpha=alpha,
                    kde=kde,
                    stat="density",
                    hue="",
                    palette=pallete,
                    ax=ax,
                )

        # Why do we need average?
        if isinstance(combined_returns, _pd.Series) or len(combined_returns.columns) == 1:
            ax.axvline(
                combined_returns.mean(),
                ls="--",
                lw=1.5,
                zorder=2,
                label="Average",
                color="orange" if font_theme == 'light' else "red",
            )

        # _plt.setp(x.get_legend().get_texts(), fontsize=11)
        ax.xaxis.set_major_formatter(
            _plt.FuncFormatter(lambda x, loc: "{:,}%".format(int(x * 100)))
        )

        # Removed static lines for clarity
        # ax.axhline(0.01, lw=1, color="#000000", zorder=2)
        # ax.axvline(0, lw=1, color="#000000", zorder=2)

        ax.set_xlabel("")
        ax.set_ylabel(
            "Occurrences", fontname=fontname, fontweight="bold", fontsize=12, color=text_color
        )
        ax.yaxis.set_label_coords(-0.1, 0.5)

        # fig.autofmt_xdate()

        try:
            _plt.subplots_adjust(hspace=0, bottom=0, top=1)
        except Exception:
            pass

        try:
            fig.tight_layout()
        except Exception:
            pass

        if savefig:
            if isinstance(savefig, dict):
                _plt.savefig(**savefig)
            else:
                _plt.savefig(savefig)

        if show:
            _plt.show(block=False)

        _plt.close()

        if not show:
            return fig

        return None


def plot_rolling_stats(
        returns,
        benchmark=None,
        title="",
        returns_label="Strategy",
        hline=None,
        hlw=None,
        hlcolor="red",
        hllabel="",
        lw=1.5,
        figsize=(10, 6),
        ylabel="",
        grayscale=False,
        background_theme=None,
        font_theme='dark',
        fontname="Arial",
        subtitle=True,
        savefig=None,
        show=True,
):
    # Grayscale and background_theme are mutually exclusive
    if grayscale:
        background_theme = None
    
    # Derive theme colors
    theme_colors = _derive_theme_colors(background_theme, font_theme)
    text_color = theme_colors['text_color']
    subtitle_color = theme_colors['subtitle_color']
    bg_color = theme_colors['bg_color']
    grid_color = theme_colors['grid_color']
    edge_color = theme_colors['edge_color']
    legend_bg = theme_colors['legend_bg']
    legend_edge = theme_colors['legend_edge']
    
    colors, _, _ = _get_colors(grayscale)

    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if isinstance(returns, _pd.DataFrame):
        returns_label = list(returns.columns)

    if isinstance(returns, _pd.Series):
        df = _pd.DataFrame(index=returns.index, data={returns_label: returns})
    elif isinstance(returns, _pd.DataFrame):
        df = _pd.DataFrame(
            index=returns.index, data={col: returns[col] for col in returns.columns}
        )
    if isinstance(benchmark, _pd.Series):
        df["Benchmark"] = benchmark[benchmark.index.isin(returns.index)]
        if isinstance(returns, _pd.Series):
            df = df[["Benchmark", returns_label]].dropna()
            ax.plot(
                df[returns_label].dropna(), lw=lw, label=returns.name, color=colors[1]
            )
        elif isinstance(returns, _pd.DataFrame):
            col_names = ["Benchmark", returns_label]
            df = df[list(_pd.core.common.flatten(col_names))].dropna()
            for i, col in enumerate(returns_label):
                ax.plot(df[col], lw=lw, label=col, color=colors[i + 1])
        ax.plot(
            df["Benchmark"], lw=lw, label=benchmark.name, color=colors[0], alpha=0.8
        )
    else:
        if isinstance(returns, _pd.Series):
            df = df[[returns_label]].dropna()
            ax.plot(
                df[returns_label].dropna(), lw=lw, label=returns.name, color=colors[1]
            )
        elif isinstance(returns, _pd.DataFrame):
            df = df[returns_label].dropna()
            for i, col in enumerate(returns_label):
                ax.plot(df[col], lw=lw, label=col, color=colors[i + 1])

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')\
    fig.suptitle(
        title, y=0.94, fontweight="bold", fontname=fontname, fontsize=14, color=text_color
    )

    if subtitle:
        ax.set_title(
            "%s - %s           \n"
            % (
                df.index.date[:1][0].strftime("%e %b '%y"),
                df.index.date[-1:][0].strftime("%e %b '%y"),
            ),
            fontsize=12,
            color=subtitle_color,
        )

    fig.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    if hline is not None:
        if not isinstance(hline, _pd.Series):
            if grayscale:
                hlcolor = "black" if background_theme is None else text_color
            ax.axhline(hline, ls="--", lw=hlw, color=hlcolor, label=hllabel, zorder=2)

    ax.axhline(0, ls="--", lw=1, color=text_color if background_theme is not None else "#000000", zorder=2)

    if ylabel:
        ax.set_ylabel(
            ylabel, fontname=fontname, fontweight="bold", fontsize=12, color=text_color
        )
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    if background_theme is not None:
        ax.tick_params(colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_color(edge_color)

    ax.yaxis.set_major_formatter(_FormatStrFormatter("%.2f"))

    legend = ax.legend(fontsize=11)
    if background_theme is not None and legend is not None:
        for text in legend.get_texts():
            text.set_color(text_color)
        # Style the legend frame for theme
        legend.get_frame().set_facecolor(legend_bg)
        legend.get_frame().set_edgecolor(legend_edge)
        legend.get_frame().set_alpha(0.8)

    if benchmark is None and len(_pd.DataFrame(returns).columns) == 1:
        ax.get_legend().remove()

    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)
    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def plot_rolling_beta(
        returns,
        benchmark,
        window1=126,
        window1_label="",
        window2=None,
        window2_label="",
        title="",
        hlcolor="red",
        figsize=(10, 6),
        grayscale=False,
        background_theme=None,
        font_theme='dark',
        fontname="Arial",
        lw=1.5,
        ylabel=True,
        subtitle=True,
        savefig=None,
        show=True,
):
    # Grayscale and background_theme are mutually exclusive
    if grayscale:
        background_theme = None
    
    # Derive theme colors
    theme_colors = _derive_theme_colors(background_theme, font_theme)
    text_color = theme_colors['text_color']
    subtitle_color = theme_colors['subtitle_color']
    bg_color = theme_colors['bg_color']
    grid_color = theme_colors['grid_color']
    edge_color = theme_colors['edge_color']
    legend_bg = theme_colors['legend_bg']
    legend_edge = theme_colors['legend_edge']
    
    colors, _, _ = _get_colors(grayscale)

    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.suptitle(
        title, y=0.94, fontweight="bold", fontname=fontname, fontsize=14, color=text_color
    )

    if subtitle:
        ax.set_title(
            "%s - %s           \n"
            % (
                returns.index.date[:1][0].strftime("%e %b '%y"),
                returns.index.date[-1:][0].strftime("%e %b '%y"),
            ),
            fontsize=12,
            color=subtitle_color,
        )

    fig.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    i = 1
    if isinstance(returns, _pd.Series):
        beta = _stats.rolling_greeks(returns, benchmark, window1)["beta"].fillna(0)
        ax.plot(beta, lw=lw, label=window1_label, color=colors[1])
    elif isinstance(returns, _pd.DataFrame):
        beta = {
            col: _stats.rolling_greeks(returns[col], benchmark, window1)["beta"].fillna(
                0
            )
            for col in returns.columns
        }
        for name, b in beta.items():
            ax.plot(b, lw=lw, label=name + " " + f"({window1_label})", color=colors[i])
            i += 1

    i = 1
    if window2:
        lw = lw - 0.5
        if isinstance(returns, _pd.Series):
            ax.plot(
                _stats.rolling_greeks(returns, benchmark, window2)["beta"],
                lw=lw,
                label=window2_label,
                color="gray",
                alpha=0.8,
            )
        elif isinstance(returns, _pd.DataFrame):
            betas_w2 = {
                col: _stats.rolling_greeks(returns[col], benchmark, window2)["beta"]
                for col in returns.columns
            }
            for name, beta_w2 in betas_w2.items():
                ax.plot(
                    beta_w2,
                    lw=lw,
                    ls="--",
                    label=name + " " + f"({window2_label})",
                    alpha=0.5,
                    color=colors[i],
                )
                i += 1

    beta_min = (
        beta.min()
        if isinstance(returns, _pd.Series)
        else min([b.min() for b in beta.values()])
    )
    beta_max = (
        beta.max()
        if isinstance(returns, _pd.Series)
        else max([b.max() for b in beta.values()])
    )
    mmin = min([-100, int(beta_min * 100)])
    mmax = max([100, int(beta_max * 100)])
    step = 50 if (mmax - mmin) >= 200 else 100
    ax.set_yticks([x / 100 for x in list(range(mmin, mmax, step))])

    if isinstance(returns, _pd.Series):
        # Keep red color for hline (consistent with other charts)
        if grayscale and background_theme is None:
            hlcolor = "black"
        ax.axhline(beta.mean(), ls="--", lw=1.5, color=hlcolor, zorder=2)

    ax.axhline(0, ls="--", lw=1, color=text_color if background_theme is not None else "#000000", zorder=2)

    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    ax.fmt_xdata = _mdates.DateFormatter("%Y-%m-%d")

    if ylabel:
        ax.set_ylabel(
            "Beta", fontname=fontname, fontweight="bold", fontsize=12, color=text_color
        )
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    if background_theme is not None:
        ax.tick_params(colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_color(edge_color)

    legend = ax.legend(fontsize=11)
    if background_theme is not None and legend is not None:
        for text in legend.get_texts():
            text.set_color(text_color)
        # Style the legend frame for theme
        legend.get_frame().set_facecolor(legend_bg)
        legend.get_frame().set_edgecolor(legend_edge)
        legend.get_frame().set_alpha(0.8)
    if benchmark is None and len(_pd.DataFrame(returns).columns) == 1:
        ax.get_legend().remove()

    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def plot_longest_drawdowns(
        returns,
        periods=5,
        lw=1.5,
        fontname="Arial",
        grayscale=False,
        background_theme=None,
        font_theme='dark',
        title=None,
        log_scale=False,
        figsize=(10, 6),
        ylabel=True,
        subtitle=True,
        compounded=True,
        savefig=None,
        show=True,
):
    # Grayscale and background_theme are mutually exclusive
    if grayscale:
        background_theme = None
    
    # Derive theme colors
    theme_colors = _derive_theme_colors(background_theme, font_theme)
    text_color = theme_colors['text_color']
    subtitle_color = theme_colors['subtitle_color']
    bg_color = theme_colors['bg_color']
    grid_color = theme_colors['grid_color']
    edge_color = theme_colors['edge_color']
    legend_bg = theme_colors['legend_bg']
    legend_edge = theme_colors['legend_edge']
    
    colors = ["#348dc1", "#003366", "red"]
    if grayscale:
        colors = ["#000000"] * 3

    dd = _stats.to_drawdown_series(returns.fillna(0))
    dddf = _stats.drawdown_details(dd)
    longest_dd = dddf.sort_values(by="days", ascending=False, kind="mergesort")[
                 :periods
                 ]

    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    
    with _ThemeContext(background_theme, font_theme):
        fig.suptitle(
            f"{title} - Worst %.0f Drawdown Periods" % periods + (" (Log Scaled)" if log_scale else ""),
            y=0.94,
            fontweight="bold",
            fontname=fontname,
            fontsize=14,
            color=text_color,
        )
        if subtitle:
            ax.set_title(
                "%s - %s           \n"
                % (
                    returns.index.date[:1][0].strftime("%e %b '%y"),
                    returns.index.date[-1:][0].strftime("%e %b '%y"),
                ),
                fontsize=12,
                color=subtitle_color,
            )

        fig.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        series = _stats.compsum(returns) if compounded else returns.cumsum()
        ax.plot(series, lw=lw, label="Backtest", color=colors[0])

        highlight = "black" if (grayscale and background_theme is None) else ("#ff6666" if font_theme == 'light' else "red")
        highlight_alpha = 0.2 if font_theme == 'light' else 0.1
        for _, row in longest_dd.iterrows():
            ax.axvspan(
                *_mdates.datestr2num([str(row["start"]), str(row["end"])]),
                color=highlight,
                alpha=highlight_alpha,
            )

        # rotate and align the tick labels so they look better
        fig.autofmt_xdate()

        # use a more precise date string for the x axis locations in the toolbar
        ax.fmt_xdata = _mdates.DateFormatter("%Y-%m-%d")

        ax.axhline(0, ls="--", lw=1, color=text_color if background_theme is not None else "#000000", zorder=2)
        _plt.yscale("symlog" if log_scale else "linear")

        # Set y-axis limits to avoid blank space at the bottom and top
        ax.set_ylim(bottom=series.min(), top=series.max())

        if ylabel:
            ax.set_ylabel(
                "Cumulative Returns",
                fontname=fontname,
                fontweight="bold",
                fontsize=12,
                color=text_color,
            )
            ax.yaxis.set_label_coords(-0.1, 0.5)
        
        if background_theme is not None:
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            for spine in ax.spines.values():
                spine.set_color(edge_color)
            ax.grid(True, color=grid_color, linewidth=0.5, alpha=1.0)

        ax.yaxis.set_major_formatter(_FuncFormatter(format_pct_axis))
        # ax.yaxis.set_major_formatter(_plt.FuncFormatter(
        #     lambda x, loc: "{:,}%".format(int(x*100))))

        fig.autofmt_xdate()

        try:
            _plt.subplots_adjust(hspace=0, bottom=0, top=1)
        except Exception:
            pass

        try:
            fig.tight_layout()
        except Exception:
            pass

        if savefig:
            if isinstance(savefig, dict):
                _plt.savefig(**savefig)
            else:
                _plt.savefig(savefig)

        if show:
            _plt.show(block=False)

        _plt.close()

        if not show:
            return fig

        return None


def plot_distribution(
        returns,
        figsize=(10, 6),
        fontname="Arial",
        grayscale=False,
        background_theme=None,
        font_theme='dark',
        ylabel=True,
        subtitle=True,
        compounded=True,
        title=None,
        savefig=None,
        show=True,
        log_scale=False,
):
    # Grayscale and background_theme are mutually exclusive
    if grayscale:
        background_theme = None
    
    # Derive theme colors
    theme_colors = _derive_theme_colors(background_theme, font_theme)
    text_color = theme_colors['text_color']
    subtitle_color = theme_colors['subtitle_color']
    bg_color = theme_colors['bg_color']
    edge_color = theme_colors['edge_color']
    
    colors = _FLATUI_COLORS
    if grayscale:
        colors = ["#f9f9f9", "#dddddd", "#bbbbbb", "#999999", "#808080"]
    # colors, ls, alpha = _get_colors(grayscale)

    port = _pd.DataFrame(returns.fillna(0))
    port.columns = ["Daily"]

    apply_fnc = _stats.comp if compounded else 'sum'

    port["Weekly"] = port["Daily"].resample("W-MON").apply(apply_fnc)
    port["Weekly"] = port["Weekly"].ffill()

    port["Monthly"] = port["Daily"].resample("ME").apply(apply_fnc)
    port["Monthly"] = port["Monthly"].ffill()

    port["Quarterly"] = port["Daily"].resample("QE").apply(apply_fnc)
    port["Quarterly"] = port["Quarterly"].ffill()

    port["Yearly"] = port["Daily"].resample("YE").apply(apply_fnc)
    port["Yearly"] = port["Yearly"].ffill()
    
    with _ThemeContext(background_theme, font_theme):
        fig, ax = _plt.subplots(figsize=figsize)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if title:
            title = f"{title} - Return Quantiles"
        else:
            title = "Return Quantiles"
        fig.suptitle(
            title, y=0.94, fontweight="bold", fontname=fontname, fontsize=14, color=text_color
        )

        if subtitle:
            ax.set_title(
                "%s - %s            \n"
                % (
                    returns.index.date[:1][0].strftime("%e %b '%y"),
                    returns.index.date[-1:][0].strftime("%e %b '%y"),
                ),
                fontsize=12,
                color=subtitle_color,
            )

        fig.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        if background_theme is not None:
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            for spine in ax.spines.values():
                spine.set_color(edge_color)

        _sns.boxplot(
            data=port,
            ax=ax,
            palette={
                "Daily": colors[0],
                "Weekly": colors[1],
                "Monthly": colors[2],
                "Quarterly": colors[3],
                "Yearly": colors[4],
            },
        )

        _plt.yscale("symlog" if log_scale else "linear")
        ax.yaxis.set_major_formatter(
            _plt.FuncFormatter(lambda x, loc: "{:,}%".format(int(x * 100)))
        )

        if ylabel:
            ax.set_ylabel(
                "Returns", fontname=fontname, fontweight="bold", fontsize=12, color=text_color
            )
            ax.yaxis.set_label_coords(-0.1, 0.5)

        fig.autofmt_xdate()

        try:
            _plt.subplots_adjust(hspace=0)
        except Exception:
            pass
        try:
            fig.tight_layout(w_pad=0, h_pad=0)
        except Exception:
            pass

        if savefig:
            if isinstance(savefig, dict):
                _plt.savefig(**savefig)
            else:
                _plt.savefig(savefig)

        if show:
            _plt.show(block=False)

        _plt.close()

        if not show:
            return fig

        return None


def plot_table(
        tbl,
        columns=None,
        title="",
        title_loc="left",
        header=True,
        colWidths=None,
        rowLoc="right",
        colLoc="right",
        colLabels=None,
        edges="horizontal",
        orient="horizontal",
        figsize=(5.5, 6),
        savefig=None,
        show=False,
):
    if columns is not None:
        try:
            tbl.columns = columns
        except Exception:
            pass

    fig = _plt.figure(figsize=figsize)
    ax = _plt.subplot(111, frame_on=False)

    if title != "":
        ax.set_title(
            title, fontweight="bold", fontsize=14, color="black", loc=title_loc
        )

    the_table = ax.table(
        cellText=tbl.values,
        colWidths=colWidths,
        rowLoc=rowLoc,
        colLoc=colLoc,
        edges=edges,
        colLabels=(tbl.columns if header else colLabels),
        loc="center",
        zorder=2,
    )

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1, 1)

    for (row, col), cell in the_table.get_celld().items():
        cell.set_height(0.08)
        cell.set_text_props(color="black")
        cell.set_edgecolor("#dddddd")
        if row == 0 and header:
            cell.set_edgecolor("black")
            cell.set_facecolor("black")
            cell.set_linewidth(2)
            cell.set_text_props(weight="bold", color="black")
        elif col == 0 and "vertical" in orient:
            cell.set_edgecolor("#dddddd")
            cell.set_linewidth(1)
            cell.set_text_props(weight="bold", color="black")
        elif row > 1:
            cell.set_linewidth(1)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    try:
        _plt.subplots_adjust(hspace=0)
    except Exception:
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def monthly_heatmap_detailedview(
        returns,
        grayscale=False,
        background_theme=None,
        font_theme='dark',
        figsize=(14, 6),
        annot_size=11,
        returns_label="Strategy",
        fontname="Arial",
        monthly_dd_font_rate=0.8,
        annual_dd_font_rate=0.8,
        return_font_rate=1.0,
        savefig=None,
        show=True,
):
    # Grayscale and background_theme are mutually exclusive
    if grayscale:
        background_theme = None
    
    # Derive theme colors
    theme_colors = _derive_theme_colors(background_theme, font_theme)
    text_color = theme_colors['text_color']
    subtitle_color = theme_colors['subtitle_color']
    bg_color = theme_colors['bg_color']
    edge_color = theme_colors['edge_color']
    
    daily_returns = returns.pct_change(fill_method=None).fillna(0)
    monthly_returns = daily_returns.resample('ME').apply(lambda x: (x + 1).prod() - 1) * 100
    monthly_drawdowns = calculate_monthly_drawdowns(returns) * 100

    monthly_combined = _pd.DataFrame({
        "Returns": monthly_returns,
        "Drawdowns": monthly_drawdowns
    })

    monthly_combined["Year"] = monthly_combined.index.year
    monthly_combined["Month"] = monthly_combined.index.month

    pivot_returns = monthly_combined.pivot(index="Year", columns="Month", values="Returns")
    pivot_drawdowns = monthly_combined.pivot(index="Year", columns="Month", values="Drawdowns")

    cmap = "gray" if grayscale else "RdYlGn"
    tick_color = text_color
    dimgray_color = subtitle_color

    fig, ax = _plt.subplots(figsize=figsize)
    ax.set_facecolor(bg_color)
    fig.set_facecolor(bg_color)

    annot_returns = pivot_returns.map(lambda x: f"{x:.2f}" if _pd.notnull(x) else "")
    annot_drawdowns = pivot_drawdowns.map(lambda x: f"{x:.2f}" if _pd.notnull(x) else "")
    mask = pivot_returns.isnull()

    return_font_size = annot_size * return_font_rate

    _sns.heatmap(
        pivot_returns,
        annot=annot_returns,
        center=0,
        annot_kws={"size": return_font_size, "ha": 'center', "va": 'bottom'},
        fmt="s",
        linewidths=0.5,
        cmap=cmap,
        cbar_kws={"format": "%.0f%%"},
        ax=ax,
        mask=mask
    )

    common_index = pivot_returns.index.intersection(annot_drawdowns.index)
    pivot_returns = pivot_returns.loc[common_index]
    annot_drawdowns = annot_drawdowns.loc[common_index]

    cell_index = 1

    for i in range(pivot_returns.shape[0]):
        for j in range(pivot_returns.shape[1]):
            if _pd.notnull(pivot_returns.iloc[i, j]):
                cell = ax.get_children()[cell_index]
                return_color = cell.get_color()

                monthly_dd_color = 'white' if return_color == 'w' else dimgray_color
                ax.text(j + 0.5, i + 0.55, annot_drawdowns.iloc[i, j],
                        ha='center', va='top', fontsize=return_font_size * monthly_dd_font_rate,
                        color=monthly_dd_color)

                cell_index += 1
            else:
                continue

    annual_returns = (pivot_returns / 100 + 1).prod(axis=1).sub(1).mul(100)
    annually_grouped = daily_returns.groupby(daily_returns.index.year)
    annual_dd = annually_grouped.apply(_stats.max_drawdown) * 100

    # Generate ytick_labels
    ytick_labels = [f"{year}\n{annual_returns[year]:.2f}" for year in pivot_returns.index]

    # Remove existing y-axis labels
    ax.set_yticks([])

    # Add new y-axis labels
    for idx, label in enumerate(ytick_labels):
        ax.text(-0.1, idx + 0.45, label,
                verticalalignment='center',
                horizontalalignment='right',
                fontsize=annot_size * 1.0,  # Maybe, 1.0 can be new argument like ytick_font_rate
                transform=ax.transData,
                color=text_color)

        # Add Drawdown
        ax.text(-0.1, idx + 0.75, f"{annual_dd[pivot_returns.index[idx]]:.2f}",
                verticalalignment='center',
                horizontalalignment='right',
                fontsize=annot_size * annual_dd_font_rate,  # Set Drawdown font size slightly smaller
                transform=ax.transData,
                color=dimgray_color)

    # Add YTD label
    ax.text(-0.1, len(pivot_returns.index) * 1.02, 'YTD', fontsize=annot_size,
            verticalalignment='center', horizontalalignment='right',
            transform=ax.transData,
            color=text_color)

    ax.set_title(
        f"{returns_label} - Monthly Returns & Drawdowns (%)",
        fontsize=14,
        fontname=fontname,
        fontweight="bold",
        color=text_color
    )

    month_abbr = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    _plt.xticks(ticks=_np.arange(0.5, 12.5, 1), labels=month_abbr, rotation=0, fontsize=annot_size)

    ax.tick_params(colors=tick_color)
    _plt.xticks(rotation=0, fontsize=annot_size * 1.2)
    _plt.yticks(rotation=0, fontsize=annot_size * 1.2)
    
    if background_theme is not None:
        # Style colorbar for theme
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.ax.tick_params(colors=text_color)
            cbar.ax.yaxis.label.set_color(text_color)

    ax.set_xlabel('')
    ax.set_ylabel('')

    _plt.tight_layout(pad=1)
    _plt.subplots_adjust(right=1.05)
    _plt.grid(False)

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def calculate_monthly_drawdowns(returns):
    drawdowns = []
    monthly_last_date = returns.resample('ME').apply(lambda x: x.index[-1]).index.tolist()
    monthly_last_trading_date = returns.resample('ME').apply(lambda x: x.index[-1]).tolist()
    monthly_last_trading_date.insert(0, returns.index[0])

    for index, end_date in enumerate(monthly_last_trading_date):
        if index == 0:
            continue

        last_month_end_date = monthly_last_trading_date[index - 1]
        current_month_returns = returns.loc[last_month_end_date:end_date]

        current_dd = _stats.max_drawdown(current_month_returns)
        drawdowns.append(current_dd)

    return _pd.Series(drawdowns, index=monthly_last_date)


def format_cur_axis(x, _):
    if x >= 1e12:
        res = "$%1.1fT" % (x * 1e-12)
        return res.replace(".0T", "T")
    if x >= 1e9:
        res = "$%1.1fB" % (x * 1e-9)
        return res.replace(".0B", "B")
    if x >= 1e6:
        res = "$%1.1fM" % (x * 1e-6)
        return res.replace(".0M", "ME")
    if x >= 1e3:
        res = "$%1.0fK" % (x * 1e-3)
        return res.replace(".0K", "K")
    res = "$%1.0f" % x
    return res.replace(".0", "")


def format_pct_axis(x, _):
    x *= 100  # lambda x, loc: "{:,}%".format(int(x * 100))
    if x >= 1e12:
        res = "%1.1fT%%" % (x * 1e-12)
        return res.replace(".0T%", "T%")
    if x >= 1e9:
        res = "%1.1fB%%" % (x * 1e-9)
        return res.replace(".0B%", "B%")
    if x >= 1e6:
        res = "%1.1fM%%" % (x * 1e-6)
        return res.replace(".0M%", "M%")
    if x >= 1e3:
        res = "%1.1fK%%" % (x * 1e-3)
        return res.replace(".0K%", "K%")
    res = "%1.0f%%" % x
    return res.replace(".0%", "%")
