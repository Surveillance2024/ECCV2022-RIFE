from InterpolatorInterface import InterpolatorInterface
interpolator = InterpolatorInterface()
result = interpolator.generate(
    imgs=('img0.png', 'img1.png'),
    exp=1
    # outputdir="interpolator_out"
)
print(result)