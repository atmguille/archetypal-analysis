"""
Module to plot the distribution of weights for a set of archetypes.

Author: Guillermo García Cobo
"""

import numpy as np
import matplotlib.pyplot as plt


def archetypal_plot(archetypes_labels: list, weights: list, filename: str = 'archetypal_plot.png'):
    """
    Plot the distribution of weights for a set of archetypes
    :param archetypes_labels: names to be plotted for each archetype
    :param weights: list of weights
    :param filename: name of the file to save the plot
    :return:
    """
    # Remove zero weights
    archetypes_labels = [a for w, a in zip(weights, archetypes_labels) if w != 0]
    weights = [w for w in weights if w != 0]
    # Compute max to normalize for plot
    weights_max = max(weights)

    n_archetypes = len(archetypes_labels)
    archetype_coordinates = np.zeros((n_archetypes, 3))

    for index, alpha in enumerate(np.arange(0.0, 2 * np.pi, 2 * np.pi / n_archetypes)):
        archetype_coordinates[index] = np.cos(alpha), np.sin(alpha), alpha

    plt.figure(figsize=(7, 7))

    ax = plt.gca()

    # Axis settings
    ax.set_xlim((-1.2, 1.2))
    ax.set_ylim((-1.2, 1.2))
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    # Eliminate axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    # Turn off tick labels
    ax.set_yticks([])
    ax.set_xticks([])

    x_coordinates, y_coordinates = [], []

    # Plot the archetypes
    for index, label in enumerate(archetypes_labels):
        x, y, alpha = archetype_coordinates[index]
        text_angle = np.rad2deg(alpha) - 90
        plt.text(x * 1.12, y * 1.12, label, ha='center', va='center', size=20, rotation=text_angle)
        # Line from centre to archetype
        plt.plot([0, x], [0, y], linestyle='--', dashes=(5, 10), linewidth=0.55, color='black')
        x, y, _ = archetype_coordinates[index] * weights[index] / weights_max
        x_coordinates.append(x)
        y_coordinates.append(y)
        plt.text(x, y, round(weights[index], 2), ha='center', va='center', size=7, rotation=text_angle,
                 color='white', fontweight='bold', bbox=dict(boxstyle=f"circle,pad=0.3", fc='black', ec='none'))

    plt.fill(x_coordinates, y_coordinates, color='r', alpha=0.2)
    plt.savefig(filename, dpi=320, bbox_inches='tight')
