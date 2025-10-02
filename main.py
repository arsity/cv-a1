import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import skimage as ski
    from matplotlib import pyplot as plt
    from scipy import signal
    return signal, mo, np, plt, ski


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Part 1""")
    return


@app.cell
def _(mo, ski):
    perspective = ski.io.imread(str(mo.notebook_location() / "public" / "perspective.png"))
    mo.image(perspective)
    return


@app.cell
def _(mo, ski):
    orthographic = ski.io.imread(str(mo.notebook_location() / "public" / "Orthographic.png"))
    mo.image(orthographic)
    return


@app.cell
def _(mo):
    mo.md(r"""Orthographic projection uses parallel lines to represent objects without a sense of depth, preserving true dimensions and shapes. In contrast, prospective projection employs converging lines, mimicking how the human eye sees by making distant objects appear smaller and creating a strong sense of realism. Orthographic views are ideal for technical drawings and measurements due to their lack of distortion and accurate representation of an object's features. Conversely, prospective projections are favored in art, computer graphics, and architecture for their ability to convey depth and a natural, photographic appearance. The key distinction lies in orthographic's dimensional accuracy versus prospective's visual realism and foreshortening.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Part 2""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## (a)""")
    return


@app.cell
def _(mo, ski, visualize_hist_cdf):
    test_img_1 = ski.io.imread(str(mo.notebook_location() / "public" / "test image 1.png"))
    eq_test_img_1 = ski.exposure.equalize_hist(test_img_1)
    eq_test_img_1 = ski.util.img_as_ubyte(eq_test_img_1)
    visualize_hist_cdf(test_img_1)
    visualize_hist_cdf(eq_test_img_1)
    return


@app.cell
def _(mo, ski, visualize_hist_cdf):
    test_img_2 = ski.io.imread(str(mo.notebook_location() / "public" / "test image 2.png"))
    eq_test_img_2 = ski.exposure.equalize_hist(test_img_2)
    eq_test_img_2 = ski.util.img_as_ubyte(eq_test_img_2)
    visualize_hist_cdf(test_img_2)
    visualize_hist_cdf(eq_test_img_2)
    return


@app.cell
def _(mo, ski, visualize_hist_cdf):
    test_img_3 = ski.io.imread(str(mo.notebook_location() / "public" / "test image 3.png"))
    eq_test_img_3 = ski.exposure.equalize_hist(test_img_3)
    eq_test_img_3 = ski.util.img_as_ubyte(eq_test_img_3)
    visualize_hist_cdf(test_img_3)
    visualize_hist_cdf(eq_test_img_3)
    return


@app.cell
def _(plt, ski):
    def visualize_hist_cdf(img):
        hist, center = ski.exposure.histogram(img)
        cdf, _ = ski.exposure.cumulative_distribution(img)

        xmin = center.min()
        xmax = center.max()

        # visualize
        fig, axs = plt.subplots(1, 2, figsize=(16, 9))
        axs[0].imshow(img, cmap="gray", vmax=255, vmin=0)
        axs[0].axis("off")
        ax1 = axs[1]
        ax2 = ax1.twinx()

        ax1.bar(center, hist, width=1, label="Histogram")
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Count", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.set_xlim(xmin, xmax)

        ax2.plot(center, cdf, color="red", label="CDF")
        ax2.set_ylabel("CDF", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.set_xlim(xmin, xmax)

        fig.tight_layout()
        plt.show()
    return (visualize_hist_cdf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## (b)""")
    return


@app.cell
def _(mo, ski, visualize_hist_cdf):
    low_contrast = ski.io.imread(str(mo.notebook_location() / "public" / "low contrast.jpeg"))
    low_contrast = ski.color.rgb2gray(low_contrast)
    low_contrast = ski.util.img_as_ubyte(low_contrast)

    eq_low_contrast = ski.exposure.equalize_hist(low_contrast)
    eq_low_contrast = ski.util.img_as_ubyte(eq_low_contrast)
    visualize_hist_cdf(low_contrast)
    visualize_hist_cdf(eq_low_contrast)
    return


@app.cell
def _(mo):
    mo.md(r"""Histogram equalization redistributes pixel intensities to create a more uniform brightness distribution across the image. This process stretches the range of intensity values, significantly increasing the global contrast, especially in areas that were initially very dark or very bright. By enhancing contrast, it aims to make hidden or subtle details more visible. I generally believe it improves image quality, particularly for images suffering from poor initial contrast, as it can reveal previously obscured information. However, it can sometimes amplify noise or produce an unnatural appearance if applied too aggressively or to images that already have good contrast.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## (d)""")
    return


@app.cell
def _(mo, plt, ski):
    test_img_4 = ski.io.imread(str(mo.notebook_location() / "public" / "test image 4.png"))
    plt.imshow(test_img_4, cmap="gray", vmin=0, vmax=255)
    return (test_img_4,)


@app.cell
def _(naive_dog, plt, test_img_4):
    fig, axs = plt.subplots(1, 2, figsize=(16, 9))
    x, y = naive_dog(test_img_4)
    axs[0].imshow(x, cmap="inferno")
    axs[1].imshow(y, cmap="inferno")
    return


@app.cell
def _(signal, np, ski):
    def naive_dog(img):
        gf_img = ski.filters.gaussian(img)

        kernel_x = np.array([[-1, 0, 1]])
        kernel_y = np.transpose(kernel_x)

        deriv_x = signal.convolve2d(gf_img, kernel_x, mode="same", boundary="symm")
        deriv_y = signal.convolve2d(gf_img, kernel_y, mode="same", boundary="symm")

        return deriv_x, deriv_y


    def direct_dog(img):
        pass
    return (naive_dog,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Part 3""")
    return


@app.cell
def _(mo, ski):
    t1 = ski.io.imread(str(mo.notebook_location() / "public" / "noisy test image 1.png"))
    t1 = ski.color.rgb2gray(t1)

    iter_slider = mo.ui.slider(start=1, stop=50, step=1)
    k_slider = mo.ui.slider(start=0.001, stop=1.0, step=0.001)

    mo.image(t1)
    return iter_slider, k_slider, t1


@app.cell
def _(iter_slider, k_slider, mo):
    mo.vstack(
        [
            mo.hstack(
                [iter_slider, mo.md(f"Iteration times: {iter_slider.value}")]
            ),
            mo.hstack([k_slider, mo.md(f"K = {k_slider.value}")]),
        ]
    )
    return


@app.cell
def _(anisotropic_diffusion, iter_slider, k_slider, mo, t1):
    out_img_1 = anisotropic_diffusion(t1, iter_slider.value, k_slider.value, 1)
    out_img_2 = anisotropic_diffusion(t1, iter_slider.value, k_slider.value, 2)
    mo.hstack(
        [
            mo.image(
                out_img_1,
                caption=f"Option 1 (niter={iter_slider.value}, k={k_slider.value})",
            ),
            mo.image(
                out_img_2,
                caption=f"Option 2 (niter={iter_slider.value}, k={k_slider.value})",
            ),
        ]
    )
    return


@app.cell
def _(mo, ski):
    t2 = ski.io.imread(str(mo.notebook_location() / "public" / "noisy test image 2.png"))

    iter_slider_2 = mo.ui.slider(start=1, stop=50, step=1)
    k_slider_2 = mo.ui.slider(start=0.001, stop=1.0, step=0.001)

    mo.image(t2)
    return iter_slider_2, k_slider_2, t2


@app.cell
def _(iter_slider_2, k_slider_2, mo):
    mo.vstack(
        [
            mo.hstack(
                [iter_slider_2, mo.md(f"Iteration times: {iter_slider_2.value}")]
            ),
            mo.hstack([k_slider_2, mo.md(f"K = {k_slider_2.value}")]),
        ]
    )
    return


@app.cell
def _(anisotropic_diffusion, iter_slider_2, k_slider_2, mo, t2):
    out_img_1_2 = anisotropic_diffusion(
        t2, iter_slider_2.value, k_slider_2.value, 1
    )
    out_img_2_2 = anisotropic_diffusion(
        t2, iter_slider_2.value, k_slider_2.value, 2
    )
    mo.hstack(
        [
            mo.image(
                out_img_1_2,
                caption=f"Option 1 (niter={iter_slider_2.value}, k={k_slider_2.value})",
            ),
            mo.image(
                out_img_2_2,
                caption=f"Option 2 (niter={iter_slider_2.value}, k={k_slider_2.value})",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""$\mathbf{K}$ controls how strong the noise we add at each step. As the iteration number increases, the image become more and more noisy.""")
    return


@app.cell
def _(np, ski):
    def anisotropic_diffusion(img, num_iter=10, k=1, option=1, gamma=0.01):
        out = ski.util.img_as_float(img)

        for _ in range(num_iter):
            padded_img = np.pad(out, 1, mode="edge")

            d_up = padded_img[0:-2, 1:-1] - out
            d_down = padded_img[2:, 1:-1] - out
            d_left = padded_img[1:-1, 0:-2] - out
            d_right = padded_img[1:-1, 2:] - out

            if option == 1:
                c_up = np.exp(-((d_up / k) ** 2))
                c_down = np.exp(-((d_down / k) ** 2))
                c_left = np.exp(-((d_left / k) ** 2))
                c_right = np.exp(-((d_right / k) ** 2))
            elif option == 2:
                c_up = 1.0 / (1.0 + (d_up / k) ** 2)
                c_down = 1.0 / (1.0 + (d_down / k) ** 2)
                c_left = 1.0 / (1.0 + (d_left / k) ** 2)
                c_right = 1.0 / (1.0 + (d_right / k) ** 2)
            else:
                raise NotImplementedError(f"Invalid option {option}")

            out += (
                c_up * d_up
                + c_down * d_down
                + c_left * d_left
                + c_right * d_right
            )

        out = out.clip(0, 1)
        return ski.util.img_as_ubyte(out)
    return (anisotropic_diffusion,)


if __name__ == "__main__":
    app.run()
