import streamlit as st
import rasterio
import numpy as np
import pandas as pd

from scipy import ndimage
from rasterio.io import MemoryFile

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="NDVI Patch Analyser",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌱 NDVI Patch Analyser")

st.write(
    """
    Upload an NDVI GeoTIFF to automatically identify vegetation
    patches and download the resulting patch raster.
    """
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("Analysis Settings")

ndvi_threshold = st.sidebar.slider(
    "NDVI threshold",
    min_value=-1.0,
    max_value=1.0,
    value=0.40,
    step=0.01,
    help="Pixels with NDVI equal to or greater than this value "
         "will be considered vegetation."
)

minimum_pixels = st.sidebar.number_input(
    "Minimum patch size (pixels)",
    min_value=1,
    value=5,
    step=1,
    help="Patches smaller than this will be removed."
)

connectivity_option = st.sidebar.selectbox(
    "Patch connectivity",
    ["8-neighbour", "4-neighbour"]
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload NDVI GeoTIFF",
    type=["tif", "tiff"]
)

# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    # --------------------------------------------------------
    # READ RASTER
    # --------------------------------------------------------

    with rasterio.open(uploaded_file) as src:

        ndvi = src.read(1).astype("float32")

        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        pixel_width = abs(transform.a)
        pixel_height = abs(transform.e)

    pixel_area = pixel_width * pixel_height

    # --------------------------------------------------------
    # DISPLAY BASIC INFORMATION
    # --------------------------------------------------------

    st.subheader("Input Raster")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", ndvi.shape[0])

    with col2:
        st.metric("Columns", ndvi.shape[1])

    with col3:
        st.metric(
            "Pixel size",
            f"{pixel_width:g} × {pixel_height:g} m"
        )

    with col4:
        if crs:
            st.metric("CRS", str(crs))
        else:
            st.metric("CRS", "Unknown")

    # --------------------------------------------------------
    # REMOVE NODATA
    # --------------------------------------------------------

    valid_pixels = np.isfinite(ndvi)

    if nodata is not None:
        valid_pixels &= ndvi != nodata

    # --------------------------------------------------------
    # CREATE VEGETATION MASK
    # --------------------------------------------------------

    vegetation = (
        valid_pixels &
        (ndvi >= ndvi_threshold)
    )

    # --------------------------------------------------------
    # CHOOSE CONNECTIVITY
    # --------------------------------------------------------

    if connectivity_option == "8-neighbour":

        structure = np.ones((3, 3), dtype=np.uint8)

    else:

        structure = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], dtype=np.uint8)

    # --------------------------------------------------------
    # IDENTIFY PATCHES
    # --------------------------------------------------------

    labels, number_of_labels = ndimage.label(
        vegetation,
        structure=structure
    )

    # --------------------------------------------------------
    # REMOVE SMALL PATCHES
    # --------------------------------------------------------

    patch_sizes = np.bincount(
        labels.ravel()
    )

    large_patch_ids = np.where(
        patch_sizes >= minimum_pixels
    )[0]

    large_patch_ids = large_patch_ids[
        large_patch_ids != 0
    ]

    # Create final patch raster
    patch_raster = np.zeros_like(
        labels,
        dtype=np.int32
    )

    new_patch_id = 1

    patch_results = []

    for old_id in large_patch_ids:

        mask = labels == old_id

        pixel_count = int(
            np.sum(mask)
        )

        # Patch area
        area_m2 = pixel_count * pixel_area
        area_ha = area_m2 / 10000

        # NDVI statistics
        values = ndvi[mask]

        mean_ndvi = float(
            np.nanmean(values)
        )

        median_ndvi = float(
            np.nanmedian(values)
        )

        min_ndvi = float(
            np.nanmin(values)
        )

        max_ndvi = float(
            np.nanmax(values)
        )

        std_ndvi = float(
            np.nanstd(values)
        )

        # Assign sequential patch ID
        patch_raster[mask] = new_patch_id

        # Quality classification
        if mean_ndvi >= 0.70:

            quality = "High"

        elif mean_ndvi >= 0.50:

            quality = "Moderate"

        else:

            quality = "Low"

        patch_results.append({

            "Patch_ID": new_patch_id,
            "Pixels": pixel_count,
            "Area_m2": round(area_m2, 2),
            "Area_ha": round(area_ha, 4),
            "Mean_NDVI": round(mean_ndvi, 4),
            "Median_NDVI": round(median_ndvi, 4),
            "Minimum_NDVI": round(min_ndvi, 4),
            "Maximum_NDVI": round(max_ndvi, 4),
            "NDVI_SD": round(std_ndvi, 4),
            "Quality": quality

        })

        new_patch_id += 1

    # --------------------------------------------------------
    # RESULTS DATAFRAME
    # --------------------------------------------------------

    results = pd.DataFrame(
        patch_results
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.subheader("Patch Summary")

    if len(results) > 0:

        total_area = results["Area_ha"].sum()

        mean_area = results["Area_ha"].mean()

        mean_ndvi = results["Mean_NDVI"].mean()

        largest_patch = results["Area_ha"].max()

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.metric(
                "Number of patches",
                len(results)
            )

        with c2:
            st.metric(
                "Total patch area",
                f"{total_area:.2f} ha"
            )

        with c3:
            st.metric(
                "Mean patch size",
                f"{mean_area:.2f} ha"
            )

        with c4:
            st.metric(
                "Largest patch",
                f"{largest_patch:.2f} ha"
            )

        with c5:
            st.metric(
                "Mean NDVI",
                f"{mean_ndvi:.3f}"
            )

        # ----------------------------------------------------
        # PATCH TABLE
        # ----------------------------------------------------

        st.subheader("Patch Statistics")

        st.dataframe(
            results,
            use_container_width=True
        )

        # ----------------------------------------------------
        # CREATE DOWNLOADABLE PATCH RASTER
        # ----------------------------------------------------

        output_profile = profile.copy()

        output_profile.update({

            "driver": "GTiff",

            "dtype": "int32",

            "count": 1,

            "nodata": 0,

            "compress": "lzw"

        })

        with MemoryFile() as memfile:

            with memfile.open(
                **output_profile
            ) as dst:

                dst.write(
                    patch_raster,
                    1
                )

            patch_tif = memfile.read()

        # ----------------------------------------------------
        # CREATE CSV
        # ----------------------------------------------------

        csv_data = results.to_csv(
            index=False
        ).encode("utf-8")

        # ----------------------------------------------------
        # DOWNLOAD BUTTONS
        # ----------------------------------------------------

        st.subheader("Download Results")

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(

                label="⬇️ Download Patch Raster",

                data=patch_tif,

                file_name="NDVI_patch_raster.tif",

                mime="image/tiff",

                use_container_width=True

            )

        with col2:

            st.download_button(

                label="⬇️ Download Patch Statistics",

                data=csv_data,

                file_name="NDVI_patch_statistics.csv",

                mime="text/csv",

                use_container_width=True

            )

        # ----------------------------------------------------
        # PATCH RASTER PREVIEW
        # ----------------------------------------------------

        st.subheader("Patch Raster Preview")

        # Mask background
        display_raster = patch_raster.astype(float)

        display_raster[
            display_raster == 0
        ] = np.nan

        st.image(
            display_raster,
            caption="Detected NDVI patches",
            use_container_width=True
        )

    else:

        st.warning(
            "No patches were detected. "
            "Try lowering the NDVI threshold or "
            "minimum patch size."
        )