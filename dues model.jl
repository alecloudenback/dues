### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# ╔═╡ 55cc95ae-0bd6-11eb-3929-2d52704266ca
begin
	using MortalityTables
	using CSV
	using DataFrames
	using Turing
	using MCMCChains
	using Plots, StatsPlots
	using MLDataUtils
	using StatsBase
end
	

# ╔═╡ 685bb6c0-0bd6-11eb-15e8-2f8882771bce
begin
	data = CSV.File("redfin current and sold.csv",normalizenames=true) |> DataFrame!
	describe(data)
end

# ╔═╡ fc102e60-0be4-11eb-2c10-b9dbf6d7c293
begin
df = data[completecases(data,
		[
			:PRICE,
			:BEDS,
			:BATHS,
			:SQUARE_FEET,
			:YEAR_BUILT,
			:HOA_MONTH,
			:NUM_UNITS,
			:Association_Heat,
			:Association_Water,
			]),:]
	filter!(row-> ~row.Other_Extras,df)
	filter!(row-> ~row.Pool,df)
	
		df.year_built_norm = let 
			x = df.YEAR_BUILT
			(x .- mean(x)) ./ std(x)
		
		
	end
	
	df
end

# ╔═╡ 58df1f50-0bf6-11eb-2b10-e123f27e2476
describe(df)

# ╔═╡ 99776f50-0bf0-11eb-13f5-71012dcc7d26
density(df[:,:HOA_MONTH])

# ╔═╡ 55013190-0be5-11eb-2e2e-8d286a5538c4
# Bayesian linear regression.
@model dues(df) = begin
    # Set variance prior.
    σ ~ Exponential(50)
    
    # Set intercept prior.
    heat_intercept ~ Normal(0, 100)
	noheat_intercept ~ Normal(0, 100)
	β_sqft ~ Normal(0,5)
	β_heat_sqft ~ Normal(0,5)
	β_units ~ Normal(0,5)
	β_age ~ Normal(0,2)
	
	for i in 1:size(df,1)
		df[i,:HOA_MONTH] ~ Normal(
			heat_intercept * df[i,:Association_Heat] +
			noheat_intercept * ~df[i,:Association_Heat] +
			β_sqft * df[i,:SQUARE_FEET] + 
			β_heat_sqft * df[i,:SQUARE_FEET] * df[i,:Association_Heat] + 
			β_units * df[i,:NUM_UNITS] + 
			β_age * df[i,:year_built_norm]
			,σ # std dev
		)
	end
    
	
end

# ╔═╡ d50abf40-0bf0-11eb-0151-21d35de1bcac
chain = sample(dues(df), NUTS(0.65), 3_000)

# ╔═╡ ff003680-0bf1-11eb-09e8-336242eb862e
corner(chain)

# ╔═╡ 30f80090-0bf3-11eb-0729-8f772548e65a
function split_data(df, target; at = 0.70)
    shuffled = shuffleobs(df)
    trainset, testset = stratifiedobs(row -> row[target], 
                                      shuffled, p = at)
end

# ╔═╡ d8701f70-0bf2-11eb-016b-cdb5e7c32ebd
begin 
	trainset,testset = split_data(df,:HOA_MONTH, at= 0.05)
	for feature in [:SQUARE_FEET,:HOA_MONTH]
  		# μ, σ = rescale!(trainset[!, feature])
  		# rescale!(testset[!, feature], μ, σ)
	end
end

# ╔═╡ a706ce30-0bfb-11eb-2cdb-15daa361b245
function posterior(chain,x)
	i = rand(1:length(chain))
	
	return rand(Normal(chain[:heat_intercept][i] * x.Association_Heat+ 
	chain[:noheat_intercept][i] * ~x.Association_Heat+ 
	x.SQUARE_FEET * chain[:β_sqft][i] +
	x.SQUARE_FEET * x.Association_Heat * chain[:β_heat_sqft][i] +
	x.NUM_UNITS * chain[:β_units][i] +
	(x.YEAR_BUILT - mean(df.YEAR_BUILT)) / std(df.YEAR_BUILT) * chain[:β_age][i]
	, chain[:σ][i]))
	
	
end


# ╔═╡ cffc17b0-0bf5-11eb-3318-bb5209ed4c1a
begin
	predictions = map(row -> [posterior(chain,row) for i in 1:1000],eachrow(df))
	
	df.predicted = mean.(predictions)
	
	df[:,[:HOA_MONTH,:SQUARE_FEET,:Association_Heat,:predicted]]
	
end
	

# ╔═╡ 064e30a0-0bf6-11eb-25c8-a983d359c77a
begin 
	@df df[df.Association_Heat .== true,:] scatter(:SQUARE_FEET,:HOA_MONTH, c=:orange,label="Heat Included",legend=:topleft,xlabel="Sq Feet",title="Monthly Dues for Condos with Heat Included and \n no extras (e.g. pool, heated garage, etc)")
	# @df df[df.Association_Heat .== false,:] scatter!(:SQUARE_FEET,:HOA_MONTH, c=:blue,label="Other Condos")
	scatter!([1445.5,1554.8],[468.41*1.01,511.06],markershape=:star5,c=:orange,markersize=10,label="Lorelei")
	
	
	
end


# ╔═╡ 6340b020-0bfc-11eb-2a66-7dc2978293f6
begin 
	bed3_dues = [posterior(chain,(
		SQUARE_FEET = 1555,
		Association_Heat = true,
		NUM_UNITS=6,
		YEAR_BUILT = 1927,
		)
) for i in 1:1000]
	p1 = density(bed3_dues)
	p2 = plot(0.01:0.01:0.99,x -> quantile(bed3_dues,x))
	plot(p1,p2)
end

# ╔═╡ 96cd099e-0c0d-11eb-26cc-279e4920b095
begin 
	scatter(df.SQUARE_FEET,df.HOA_MONTH .- df.predicted, c=ifelse.(df.Association_Heat, :orange, :blue), title = residuals)
	
	
	
end


# ╔═╡ Cell order:
# ╠═55cc95ae-0bd6-11eb-3929-2d52704266ca
# ╠═685bb6c0-0bd6-11eb-15e8-2f8882771bce
# ╠═fc102e60-0be4-11eb-2c10-b9dbf6d7c293
# ╠═58df1f50-0bf6-11eb-2b10-e123f27e2476
# ╠═99776f50-0bf0-11eb-13f5-71012dcc7d26
# ╠═d8701f70-0bf2-11eb-016b-cdb5e7c32ebd
# ╠═55013190-0be5-11eb-2e2e-8d286a5538c4
# ╠═d50abf40-0bf0-11eb-0151-21d35de1bcac
# ╠═ff003680-0bf1-11eb-09e8-336242eb862e
# ╠═30f80090-0bf3-11eb-0729-8f772548e65a
# ╠═a706ce30-0bfb-11eb-2cdb-15daa361b245
# ╠═cffc17b0-0bf5-11eb-3318-bb5209ed4c1a
# ╠═064e30a0-0bf6-11eb-25c8-a983d359c77a
# ╠═6340b020-0bfc-11eb-2a66-7dc2978293f6
# ╠═96cd099e-0c0d-11eb-26cc-279e4920b095
