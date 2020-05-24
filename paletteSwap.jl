using LinearAlgebra, Images, FileIO, Colors

function sinkhorn( a, b, precision, costMatrix, n_iter )
	n, m = length( a ), length( b )
	C = costMatrix
	costMatrix = costMatrix ./ max( costMatrix... ) # renormalizing to avoid underflows in division
	K = exp.( -costMatrix/precision )
	u, v = ones( n ), ones( m )
	for i in 1:n_iter
		# u = a ./ K*v
		u = exp.( log.( a ) - log.( K*v ) )
		v = exp.( log.( b ) - log.( K'*u ) )
	end
	# P = [ u[i]*K[i,j]*v[j] for i in 1:n, j in 1:m ]
	P = u.*K*diagm( v )
	entropy = -sum( P.*log.( P ) )
	return P, sum( P.*C ) - precision*entropy
end

function getImage( path::String, quantization::Int )
	execstring =`convert $path -dither None -colors $quantization $path.temp.png`
	run(execstring)
	img = load( "$path.temp.png" )
	rm( "$path.temp.png" )
	return img
end

function swapPalette( paletteSourceImage::String, paletteTargetImage::String, maxColors=10_000, mode="mean" )
	source = RGB.( getImage( paletteSourceImage, maxColors ) )
	target = RGB.( getImage( paletteTargetImage, maxColors ) )

	scols = unique( source )
	tcols = unique( target )

	# we don't account for frequency of colors, so the distributions are uniform
	scdist = ones( length( scols ) )/ length( scols )
	tcdist = ones( length( tcols ) )/ length( tcols )

	costMatrix = [ colordiff( t, s ) for t in tcols, s in scols ]

	P, d = sinkhorn( tcdist, scdist, 0.01, costMatrix, 100 )

	if mode=="matching"
		perm = [ argmax( P[i,:] ) for i in 1:length( tcols ) ]
		mp = [ tcols[i] => scols[perm][i] for i in 1:length( tcols ) ]
	elseif mode=="mean"
		scols = LCHab.( scols )
		scols = [ [a.l, a.c, a.h] for a in scols ]
		mp = [tcols[i] => LCHab( sum( scols.*P[i,:]/sum( P[i,:] ) )... ) for i in 1:length( tcols )]
	else
		@error "Unknown mode:\t$mode."
	end
	for i in 1:length( tcols )  replace!( target, mp[i] )  end
	return target
end

nargs = length( ARGS )
if 3 <= nargs<= 5
	maxColors = nargs>=4 ? ARGS[4] : 10_000
	mode      = nargs==5 ? ARGS[5] : "mean"
	@time img = swapPalette( ARGS[1], ARGS[2], maxColors, mode )
	save( ARGS[3], img )
else
	println( "Usage: julia paletteSwap.jl paletteSourceImage paletteTargetImage outputFile [maxColors=10_000 mode='mean']" )
end
