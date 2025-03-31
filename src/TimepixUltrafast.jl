module TimepixUltrafast

using DataFrames
using Distributions
using StatsBase
import Base.Threads.@threads

export  estimate_signal,
        ei_car_coin_background,
        ei_car_coin_measurement,
        ei_car_corr_coin,
        ei_car_kovariance,
        ei_r_coin_background,
        ei_r_coin_measurement,
        ei_r_corr_coin,
        ei_r_kovariance

include("electron_ion.jl")

end

