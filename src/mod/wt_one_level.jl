# ========== Perform 1 step of discrete wavelet transform ==========
# ----- 1 step of dwt for 2D signals -----
function dwt_step!(y::AbstractArray{T,2},
                   x::AbstractArray{T,2},
                   filter::OrthoFilter,
                   dcfilter::StridedVector{S},
                   scfilter::StridedVector{S},
                   si::StridedVector{S};
                   standard::Bool = true) where {T<:Number, S<:Number}
    # Sanity check
    @assert size(x) == size(y)
    # Is forward transform
    fw = true
    # Temporary matrix
    temp = similar(x)
    if standard
        # Compute dwt for all rows
        @views for (tempᵢ, xᵢ) in zip(eachrow(temp), eachrow(x))
            Transforms.unsafe_dwt1level!(tempᵢ, xᵢ, filter, fw, dcfilter, scfilter, si)
        end
        # Compute dwt for all columns
        @views for (yⱼ, tempⱼ) in zip(eachcol(y), eachcol(temp))
            Transforms.unsafe_dwt1level!(yⱼ, tempⱼ, filter, fw, dcfilter, scfilter, si)
        end
    else
        # TODO: Implement non-standard transform
        error("Non-standard transform not implemented yet.")
    end
    return y
end
