# Table of contents
- [COLORIZE](#colorize)
  * [Dataset](#dataset)
  * [Modelo](#modelo)
  * [Generador](#generador)
  * [Discriminador](#discriminador)
  * [Loss Function](#loss-function)
  * [Entrenamiento](#entrenamiento)
    + [Bucle de entrenamiento:](#bucle-de-entrenamiento-)




# COLORIZE
Base para el entrenamiento de una red neuronal a la conversion de imágenes en blanco y negro a imágenes en color.

## Dataset
Se trabajara con un conjunto de unas 1200 imágenes locales a color, tamaño variado y formato jpg. Estas serán separadas en un 90% entrenamiento y un 10% test (para comprobar como la red neuronal maneja imágenes que nunca ha visto). Las imágenes se reescalan al inicio a un tamaño cuadrado fijo de 120x120 pixeles para poder procesarse.

```
# Nuestra data 
master_dir = "classification/test/24"       # carpeta 1229 imagenes 
IMG_SIZE = 120                              # nuevo tamaño de 120x120

print("Contenido total de imagenes: ",len(os.listdir( master_dir ))) 
```
> Contenido total de imagenes:  1229



Para reducir los tiempos de procesado se limitan la cantidad de imagenes a 200. Se almacenan dos listas con las imágenes en blanco y negro (x) y las de color (y). Serán almacenadas normalizadas dado que las redes neuronales obtienen mejores resultados entre valores de 0 a 1.
```
x = []
y = []
for image_file in os.listdir( master_dir )[ 0 : 200 ]:
    ...

print("Imagenes B&W: ", len(x))
print("Imagenes a color: ", len(y))
```
> Imagenes B&W:  200
>
> Imagenes a color :  200

Estas imagenes seleccionadas se tienen que separar en las de entrenamiento y de test para comprobar los resultados reales de la red neuronal y que no se confundan con los resultados del entrenamiento.

```
# Train-test splitting
train_x, test_x, train_y, test_y = train_test_split( np.array(x) , np.array(y) , test_size=0.1 ) # Separar imagenes en entrenamiento y test

# Construct tf.data.Dataset object
dataset = tf.data.Dataset.from_tensor_slices( ( train_x , train_y ) )
dataset = dataset.batch( 30 )

print("entrenamiento: ", len(train_x))
print("test: ", len(test_x))
print(f"valores normalizados: máximo: {np.max(train_x)}, mínimo: {np.min(train_x)}")
```

> entrenamiento:  180
>
> test:  20
>
> valores normalizados: máximo: 1.0, mínimo: 0.0

## Modelo
El modelo es el conjunto de funciones que resultan en una red neuronal. Una estructura basica es la que se empleará en este modelo.
* Generador: Crea una nueva matriz y a partir de x
* Discriminador: Guarda los cambios de cada red neuronal que mejoran el resultado.
* Loss Function: Decide que resultados estan MENOS MAL.


## Generador
El modelo recibirá una matriz de (120,120,1) para obtener una matriz de (120,120,3)

```
def get_generator_model():

    inputs = tf.keras.layers.Input( shape=( IMG_SIZE , IMG_SIZE , 1 ) )

    conv1 = tf.keras.layers.Conv2D( 16 , kernel_size=( 5 , 5 ) , strides=1 )( inputs )
    conv1 = tf.keras.layers.LeakyReLU()( conv1 )
    conv1 = tf.keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1)( conv1 )
    conv1 = tf.keras.layers.LeakyReLU()( conv1 )
    conv1 = tf.keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1)( conv1 )
    conv1 = tf.keras.layers.LeakyReLU()( conv1 )

    conv2 = tf.keras.layers.Conv2D( 32 , kernel_size=( 5 , 5 ) , strides=1)( conv1 )
    conv2 = tf.keras.layers.LeakyReLU()( conv2 )
    conv2 = tf.keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 )( conv2 )
    conv2 = tf.keras.layers.LeakyReLU()( conv2 )
    conv2 = tf.keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 )( conv2 )
    conv2 = tf.keras.layers.LeakyReLU()( conv2 )

    conv3 = tf.keras.layers.Conv2D( 64 , kernel_size=( 5 , 5 ) , strides=1 )( conv2 )
    conv3 = tf.keras.layers.LeakyReLU()( conv3 )
    conv3 = tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 )( conv3 )
    conv3 = tf.keras.layers.LeakyReLU()( conv3 )
    conv3 = tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 )( conv3 )
    conv3 = tf.keras.layers.LeakyReLU()( conv3 )

    bottleneck = tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='tanh' , padding='same' )( conv3 )

    concat_1 = tf.keras.layers.Concatenate()( [ bottleneck , conv3 ] )
    conv_up_3 = tf.keras.layers.Conv2DTranspose( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' )( concat_1 )
    conv_up_3 = tf.keras.layers.Conv2DTranspose( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' )( conv_up_3 )
    conv_up_3 = tf.keras.layers.Conv2DTranspose( 64 , kernel_size=( 5 , 5 ) , strides=1 , activation='relu' )( conv_up_3 )

    concat_2 = tf.keras.layers.Concatenate()( [ conv_up_3 , conv2 ] )
    conv_up_2 = tf.keras.layers.Conv2DTranspose( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' )( concat_2 )
    conv_up_2 = tf.keras.layers.Conv2DTranspose( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' )( conv_up_2 )
    conv_up_2 = tf.keras.layers.Conv2DTranspose( 32 , kernel_size=( 5 , 5 ) , strides=1 , activation='relu' )( conv_up_2 )

    concat_3 = tf.keras.layers.Concatenate()( [ conv_up_2 , conv1 ] )
    conv_up_1 = tf.keras.layers.Conv2DTranspose( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu')( concat_3 )
    conv_up_1 = tf.keras.layers.Conv2DTranspose( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu')( conv_up_1 )
    conv_up_1 = tf.keras.layers.Conv2DTranspose( 3 , kernel_size=( 5 , 5 ) , strides=1 , activation='relu')( conv_up_1 )

    model = tf.keras.models.Model( inputs , conv_up_1 )
    return model

generator = get_generator_model()
```
## Discriminador
El discriminador se encargará de guardar los cambios en los weights y bias de cada neurona que devuelvan un menor error respecto de el "supusesto output" que llamamos train_y.

```
def get_discriminator_model():
    layers = [
        tf.keras.layers.Conv2D( 32 , kernel_size=( 7 , 7 ) , strides=1 , activation='relu' , input_shape=( 120 , 120 , 3 ) ),
        tf.keras.layers.Conv2D( 32 , kernel_size=( 7, 7 ) , strides=1, activation='relu'  ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D( 64 , kernel_size=( 5 , 5 ) , strides=1, activation='relu'  ),
        tf.keras.layers.Conv2D( 64 , kernel_size=( 5 , 5 ) , strides=1, activation='relu'  ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1, activation='relu'  ),
        tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1, activation='relu'  ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1, activation='relu'  ),
        tf.keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1, activation='relu'  ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense( 512, activation='relu'  )  ,
        tf.keras.layers.Dense( 128 , activation='relu' ) ,
        tf.keras.layers.Dense( 16 , activation='relu' ) ,
        tf.keras.layers.Dense( 1 , activation='sigmoid' ) 
    ]
    model = tf.keras.models.Sequential( layers )
    return model

discriminator = get_discriminator_model()
```
## Loss Function

```
cross_entropy = tf.keras.losses.BinaryCrossentropy()
mse = tf.keras.losses.MeanSquaredError()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) - tf.random.uniform( shape=real_output.shape , maxval=0.1 ) , real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output) + tf.random.uniform( shape=fake_output.shape , maxval=0.1  ) , fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output , real_y):
    real_y = tf.cast( real_y , 'float32' )
    return mse( fake_output , real_y )

generator_optimizer = tf.keras.optimizers.Adam( 0.0005 )
discriminator_optimizer = tf.keras.optimizers.Adam( 0.0005 )

generator = get_generator_model()
discriminator = get_discriminator_model()
```

## Entrenamiento
Funcion de entrenamiento:

```
def train_step( input_x , real_y ):
   
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate an image -> G( x )
        generated_images = generator( input_x , training=True)
        # Probability that the given image is real -> D( x )
        real_output = discriminator( real_y, training=True)
        # Probability that the given image is the one generated -> D( G( x ) )
        generated_output = discriminator(generated_images, training=True)
        
        # L2 Loss -> || y - G(x) ||^2
        gen_loss = generator_loss( generated_images , real_y )
        # Log loss for the discriminator
        disc_loss = discriminator_loss( real_output, generated_output )
        
        losses["D"].append(disc_loss.numpy())
        losses["G"].append(gen_loss.numpy())
    #tf.keras.backend.print_tensor( tf.keras.backend.mean( gen_loss ) )
    #tf.keras.backend.print_tensor( gen_loss + disc_loss )
    
    # Compute the gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Optimize with Adam
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

generator.compile(
    optimizer=generator_optimizer,
    loss=generator_loss,
    metrics=['accuracy']
)

discriminator.compile(
    optimizer=discriminator_optimizer,
    loss=discriminator_loss,
    metrics=['accuracy']
)
```

### Bucle de entrenamiento:
Para 5 epochs (repeticiones de cada red neuronal). Aumentar este valor indiscriminadamente no devuelve un mejor resultado y puede generar overfitting donde la red neuronal memoriza los patrones de la informacion dada por el usuario y tiene muy malos resultados en datos que no ha visto antes.
```
num_epochs = 5
losses = {"D":[], "G":[]}
for e in range( num_epochs ):
    print("Running epoch : ", e )
    for ( x , y ) in dataset:
        train_step( x ,y )
```

![alt text](fig/colorize_output.png)


## Otras adiciones
Las redes neuronales calculan los resultados como probabilidades que suelen aumentar en funcion de las repeticiones. Una forma de evitar el overfitting que puede ocurrir es introducir una funcion callback que automatiza el numero de epochs que se realizarán.

```
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.80):
      print("\nReached 80% accuracy so cancelling training!")
      self.model.stop_training = True
```





