# CODIGO PRA AUMENTAR A BASE DE IMAGENS
# FUNCIONANDO PERFEITAMENTE

import Augmentor

p = Augmentor.Pipeline("/home/mpierre/PycharmProjects/ImgArmadilhas/BaseOutubro/ameloblastoma")

p.rotate90(probability=0.5)
p.rotate270(probability=0.5)
p.rotate180(probability=0.2)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
#p.crop_random(probability=1, percentage_area=0.5)

p.resize(probability=1.0, width=299, height=299)

# Here we sample 100,000 images from the pipeline.
# It is often useful to use scientific notation for specify
# large numbers with trailing zeros.
# 100.000 imagens geradas
# num_of_samples = int(1e5)

num_of_samples = 2500

# Now we can sample from the pipeline:
p.sample(num_of_samples)

