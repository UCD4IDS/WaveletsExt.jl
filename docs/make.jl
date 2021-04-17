using Documenter, WaveletsExt

makedocs(
    sitename = "WaveletsExt.jl",
    format = Documenter.HTML(),
    authors = "Zeng Fung Liew",
    clean = true,
    pages = [
        "index.md",
        "transforms.md",
        "bestbasis.md",
        "denoising.md",
        "api.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/zengfung/WaveletsExt.jl.git"
)
