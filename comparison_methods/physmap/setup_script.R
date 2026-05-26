# Setup script for the PhysMAP R environment.
#
# Reviewers / first-time users: prefer
#
#   R -e 'renv::restore()'
#
# which reads the bundled renv.lock and installs the exact package versions
# used to produce the paper figures (R 4.2.1, Seurat 4.3.0, dplyr 1.1.2, ...).
#
# Run THIS script only when you want to refresh the lockfile against newer
# package versions. physmap_script.r uses Seurat v4 APIs (CreateSeuratObject +
# assay-based slots) and breaks under Seurat v5, so we cap Seurat below 5.0.

# Install renv if not already installed (to user library)
if (!requireNamespace("renv", quietly = TRUE)) {
    install.packages("renv", lib = Sys.getenv("R_LIBS_USER"))
}

renv::init()

# Explicit dependency set. Adjust versions only after re-validating that
# physmap_script.r still runs end-to-end on the new combination.
packages_to_install <- c(
    "tidyr",
    "dplyr",
    "caret",
    "nnet",
    "reshape2",
    "aricode",
    "ggplot2",
    "patchwork",
    "jsonlite",
    "Seurat@<5.0"   # v4 API required by physmap_script.r
)

renv::install(packages_to_install)
renv::snapshot()

# Verify installations
for (pkg in packages_to_install) {
    if (require(pkg, character.only = TRUE, quietly = TRUE)) {
        cat(paste("✓", pkg, "installed successfully\n"))
    } else {
        cat(paste("✗", pkg, "failed to install\n"))
    }
}

cat("\nEnvironment setup complete!")
cat("\nTo restore this environment later, run: renv::restore()")
cat("\nTo update packages, run: renv::update()")
cat("\nTo create a snapshot, run: renv::snapshot()")
