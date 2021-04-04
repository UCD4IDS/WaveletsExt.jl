using Documenter
using WaveletsExt

makedocs(
    sitename = "WaveletsExt.jl",
    format = Documenter.HTML(),
    authors = "Zeng Fung Liew",
    modules = [WaveletsExt]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/zengfung/WaveletsExt.jl"
)
