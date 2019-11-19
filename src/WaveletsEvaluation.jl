module WaveletsEvaluation

using RecipesBase, Reexport, CardinalBSplines, InfiniteVectors


include("Embeddings.jl/Embeddings.jl")
@reexport using .Embeddings

include("filterbanks/filterbank.jl")
include("dwt/discretewavelets.jl")

@reexport using .DWT

end
