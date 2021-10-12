ENV["GKSwstype"] = "100"
ENV["PLOTS_TEST"] = "true"
using Documenter, Wavelets, WaveletsExt

makedocs(
    sitename = "WaveletsExt.jl",
    format = Documenter.HTML(),
    authors = "Zeng Fung Liew, Shozen Dan",
    clean = true,
    pages = Any[
        "Home" => "index.md",
        "Manual" => Any[
            "Transforms" => "manual/transforms.md",
            "Best Basis" => "manual/bestbasis.md",
            "Denoising" => "manual/denoising.md",
            "Local Discriminant Basis" => "manual/localdiscriminantbasis.md"
        ],
        "API" => Any[
            "DWT" => "api/dwt.md",
            "ACWT" => "api/acwt.md",
            "SWT" => "api/swt.md",
            "TIWT" => "api/tiwt.md",
            "Best Basis" => "api/bestbasis.md",
            "Denoising" => "api/denoising.md",
            "LDB" => "api/ldb.md",
            "SIWPD" => "api/siwpd.md",
            "Utils" => "api/utils.md",
            "Visualizations" => "api/visualizations.md",
        ]
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/UCD4IDS/WaveletsExt.jl.git"
)
