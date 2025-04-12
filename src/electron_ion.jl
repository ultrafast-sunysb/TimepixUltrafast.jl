@doc raw"""
    estimate_signal(bg_dist::UnivariateDistribution, meas::Integer)

Computes, for `meas` (``M``) measured counts, and `bg_dist` (``B``) background
distribution,

```math
\sum_{b=0}^M (M - b) \frac{pdf(B, b)}{cdf(B, M)}
= M - \sum_{b=0}^M b \frac{pdf(B, b)}{cdf(B, M)}
```
"""
function estimate_signal(bg_dist::UnivariateDistribution, meas_counts::Integer)
    acc::Float64 = 0
    for b in 0:meas_counts
        acc += (meas_counts - b) * pdf(bg_dist, b)
    end
    return acc / cdf(bg_dist, meas_counts)
end


# iterate over electron hits *without* ion hits to calculate the number of shots
# without ion, and histogram of these electron hits
function ei_car_coin_background(ele::DataFrame, ion::DataFrame,
    bin_range = (0:255, 0:255)
)::Tuple{UInt64, Matrix{UInt64}}
    background = zeros(UInt64, size(bin_range[1])[1], size(bin_range[2])[1])
    shots_without_ion = 0
    last_shot = -1
    for ele_hit_idx in 1:size(ele)[1]
        if !insorted(ele[ele_hit_idx, :shot], ion.shot)
            x = searchsortedlast(bin_range[1], ele[ele_hit_idx, :x])
            y = searchsortedlast(bin_range[2], ele[ele_hit_idx, :y])
            background[x, y] += 1
            if ele[ele_hit_idx, :shot] > last_shot
                last_shot = ele[ele_hit_idx, :shot]
                shots_without_ion += 1
            end
        end
    end
    return (shots_without_ion, background)
end


# iterate over electron hits *with* ion hits to calculate the number of shots
# withion, and histogram of these electron hits
function ei_car_coin_measurement(ele::DataFrame, ion::DataFrame,
    bin_range = (0:255, 0:255)
)::Tuple{UInt64, Matrix{UInt64}}
    measurement = zeros(UInt64, size(bin_range[1])[1], size(bin_range[2])[1])
    # iterate over ion hits and count corrected coincidences
    shots_with_ion = 0
    last_shot = -1
    for ion_hit_idx in 1:size(ion)[1]
        # get indices of electron hits (assumes shot index is monotonic)
        ele_hit_idxs = searchsorted(ele.shot, ion[ion_hit_idx, :shot])
        if ele_hit_idxs.stop >= ele_hit_idxs.start
            for ele_hit_idx in ele_hit_idxs
                x = searchsortedlast(bin_range[1], ele[ele_hit_idx, :x])
                y = searchsortedlast(bin_range[2], ele[ele_hit_idx, :y])
                measurement[x, y] += 1
            end
            if ion[ion_hit_idx, :shot] > last_shot
                last_shot = ion[ion_hit_idx, :shot]
                shots_with_ion += 1
            end
        end
    end
    return (shots_with_ion, measurement)
end


# electron-ion corrected coincidences in cartesian coordinates
# returns corrected coincidences of electrons at x,y with number of ions
function ei_car_corr_coin(ele::DataFrame, ion::DataFrame;
    bin_range = (0:255, 0:255), simple_bg = false
)::Matrix
    # background electron hits
    bg_task = @spawn ei_car_coin_background(ele, ion, bin_range)
    # sum of measured electron hits
    meas_task = @spawn ei_car_coin_measurement(ele, ion, bin_range)
    # join threads
    bg_shots, bg = fetch(bg_task)
    meas_shots, meas = fetch(meas_task)
    # now "subtract" the background
    if simple_bg
        return (
            (meas - bg * (meas_shots / bg_shots))
            ./ (ele.shot[end] - ele.shot[1] + 1)
        )
    else
        # (poisson and binomial are about the same, but poisson will allow you
        # to have more than 1 hit per shot)
        #return estimate_signal.(Binomial.(meas_shots, bg ./ bg_shots), meas)
        #return estimate_signal.(Poisson.(bg .* (meas_shots / bg_shots)), meas)
        return (
            estimate_signal.(Poisson.(bg .* (meas_shots / bg_shots)), meas)
            ./ (ele.shot[end] - ele.shot[1] + 1)
        )
    end
end


# kovi style
function ei_car_kovariance(ele::DataFrame, ion::DataFrame;
    bin_range = (0:255, 0:255)
)::Matrix
    edges = (
        (bin_range[1][1]):(step(bin_range[1])):(
            bin_range[1][end] + step(bin_range[1])
        ),
        (bin_range[2][1]):(step(bin_range[2])):(
            bin_range[2][end] + step(bin_range[2])
        ),
    )
    bg_hist = fit(Histogram, (ele.x, ele.y), edges)
    meas_shot_hist = fit(Histogram, ion.shot, ion.shot[1]:ion.shot[end])
    meas_shots, meas = ei_car_coin_measurement(ele, ion, bin_range)
    return clamp.(
        meas / sum(meas_shot_hist.weights)
        - bg_hist.weights / length(unique(ele.shot)),
        0, Inf
    )
end


function ei_r_coin_background(ele::DataFrame, ion::DataFrame, bin_range = 0:127
)::Tuple{UInt64, Vector{UInt64}}
    background = zeros(UInt64, size(bin_range)[1])
    shots_without_ion = 0
    last_shot = -1
    for ele_hit_idx in 1:size(ele)[1]
        if !insorted(ele[ele_hit_idx, :shot], ion.shot)
            background[searchsortedlast(bin_range, ele[ele_hit_idx, :r])] += 1
            if ele[ele_hit_idx, :shot] > last_shot
                last_shot = ele[ele_hit_idx, :shot]
                shots_without_ion += 1
            end
        end
    end
    return (shots_without_ion, background)
end


function ei_r_coin_measurement(ele::DataFrame, ion::DataFrame, bin_range = 0:127
)::Tuple{UInt64, Vector{UInt64}}
    measurement = zeros(UInt64, size(bin_range)[1])
    # iterate over ion hits and count corrected coincidences
    shots_with_ion = 0
    last_shot = -1
    for ion_hit_idx in 1:size(ion)[1]
        # get indices of electron hits (assumes shot index is monotonic)
        ele_hit_idxs = searchsorted(ele.shot, ion[ion_hit_idx, :shot])
        if ele_hit_idxs.stop >= ele_hit_idxs.start
            for ele_hit_idx in ele_hit_idxs
                measurement[
                    searchsortedlast(bin_range, ele[ele_hit_idx, :r])
                ] += 1
            end
            if ion[ion_hit_idx, :shot] > last_shot
                last_shot = ion[ion_hit_idx, :shot]
                shots_with_ion += 1
            end
        end
    end
    return (shots_with_ion, measurement)
end


# electron-ion corrected coincidences in radius
# returns corrected coincidences of electrons at r with number of ions
function ei_r_corr_coin(ele::DataFrame, ion::DataFrame;
    bin_range = 0:127, simple_bg = false
)::Vector
    bg_task = @spawn ei_r_coin_background(ele, ion, bin_range)
    meas_task = @spawn ei_r_coin_measurement(ele, ion, bin_range)
    bg_shots, bg = fetch(bg_task)
    meas_shots, meas = fetch(meas_task)
    if simple_bg
        return (
            (meas - bg * meas_shots / bg_shots)
            ./ (ele.shot[end] - ele.shot[1] + 1)
        )
    else
        #return estimate_signal.(Poisson.(bg .* (meas_shots / bg_shots)), meas)
        return (
            estimate_signal.(Poisson.(bg .* (meas_shots / bg_shots)), meas)
            ./ (ele.shot[end] - ele.shot[1] + 1)
        )
    end
end


# kovi style
function ei_r_kovariance(ele::DataFrame, ion::DataFrame; bin_range = 0:127
)::Vector
    edges = (bin_range[1]):(step(bin_range)):(bin_range[end] + step(bin_range))
    bg_hist = fit(Histogram, ele.r, edges)
    meas_shot_hist = fit(Histogram, ion.shot, ion.shot[1]:ion.shot[end])
    meas_shots, meas = ei_r_coin_measurement(ele, ion, bin_range)
    return (
        meas / sum(meas_shot_hist.weights)
        - bg_hist.weights / length(unique(ele.shot))
    )
end

