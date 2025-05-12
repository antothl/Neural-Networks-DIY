import numpy as np

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        if self._gradient is not None:
            self._gradient=np.zeros_like(self._parameters)

    def forward(self, X):
        raise NotImplementedError

    def update_parameters(self, gradient_step=1e-3):
        if self._parameters is not None and self._gradient is not None:
            self._parameters -= gradient_step * self._gradient

    def backward_update_gradient(self, input, delta):
        raise NotImplementedError

    def backward_delta(self, input, delta):
        raise NotImplementedError

class Linear(Module):
    def __init__(self, input_dim, output_dim, init=None):
        super().__init__()
        if init is not None:
            self._parameters = init(input_dim, output_dim)
        else:
            self._parameters = np.random.randn(input_dim + 1, output_dim)
        self._gradient = np.zeros((input_dim + 1, output_dim))

    def forward(self, X):
        X = np.hstack([X, np.ones((X.shape[0], 1))])  
        return X @ self._parameters  

    def backward_update_gradient(self, input, delta):
        input = np.hstack([input, np.ones((input.shape[0], 1))])                  
        if delta.ndim == 1:
            delta = delta.reshape(-1, 1)
        self._gradient += (input.T @ delta)

    def backward_delta(self, input, delta):
        if delta.ndim == 1:
            delta = delta.reshape(-1, 1)

        return delta @ self._parameters[:-1].T  
    
class TanH(Module):
    def forward(self, X):
        self.X = X
        return np.tanh(X)

    def backward_delta(self, X, delta):
        if delta.ndim == 1:
            delta = delta.reshape(-1, 1)
        return delta * (1 - np.tanh(X) ** 2)  
    
    def backward_update_gradient(self, X, delta):
        pass 
    
    def update_parameters(self, lr):
        pass  

class Sigmoide(Module):
    def forward(self, X):
        self.output = 1 / (1 + np.exp(-X))
        return self.output

    def backward_delta(self, X, delta):
        return delta * self.output * (1 - self.output) 
    
    def backward_update_gradient(self, X, delta):
        pass  
    
    def update_parameters(self, lr):
        pass  

class SimpleNN(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.linear1 = Linear(input_dim, hidden_dim)
        self.tanh = TanH()
        self.linear2 = Linear(hidden_dim, output_dim)
        self.sigmoid = Sigmoide()

    def forward(self, X):
        self.z1 = self.linear1.forward(X)     
        self.a1 = self.tanh.forward(self.z1)  
        self.z2 = self.linear2.forward(self.a1)
        y_hat = self.sigmoid.forward(self.z2)
    
        return y_hat

    def backward(self, X, y, loss):
        y_hat = self.forward(X)
        delta = loss.backward(y, y_hat)

        delta2 = self.sigmoid.backward_delta(self.z2, delta)  
        delta1 = self.linear2.backward_delta(self.a1, delta2)
        delta0 = self.tanh.backward_delta(self.z1, delta1) 

        self.linear2.backward_update_gradient(self.a1, delta2)
        self.linear1.backward_update_gradient(X, delta0)

    def update_parameters(self, lr):
        self.linear1.update_parameters(lr)
        self.linear2.update_parameters(lr)

    def zero_grad(self):
        self.linear1.zero_grad()
        self.linear2.zero_grad()


class Sequentiel_old(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = []
        for m in modules:
            if isinstance(m, Sequentiel_old):
                self.modules.extend(m.modules)
            else:
                self.modules.append(m)
    
    def forward(self, X):
        self.inputs = []  
        for module in self.modules:
            self.inputs.append(X)  
            X = module.forward(X)
        return X

    def backward(self, X, y, loss):
        y_hat = self.forward(X)
        delta = loss.backward(y, y_hat)
        
        for i in range(len(self.modules)-1, -1, -1):
            module = self.modules[i]
            input = self.inputs[i] if i > 0 else X
            
            module.backward_update_gradient(input, delta)
            delta = module.backward_delta(input, delta)


    def update_parameters(self, lr):
        for module in self.modules:
            module.update_parameters(lr)
    
    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

class Sequentiel(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        self.inputs = []  # pour stocker les entrées de chaque couche
        self._debug = False

    def forward(self, x):
        self.inputs = [x]
        for layer in self.layers:
            if self._debug:
                print(layer.__class__.__name__, x.shape, '(batch size, input_dim)')
            x = layer.forward(x)
            if self._debug:
                print('output shape:', x.shape)
            self.inputs.append(x)  # on garde les entrées de chaque couche pour la rétropropagation
        if self._debug:
            print('Forward pass done\n----------------')
        return x
    
    def backward_update_gradient(self, input, delta):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            input_i = self.inputs[i]
            if self._debug:
                print(layer.__class__.__name__, '(batch size, input_dim)')
                print('delta l+1 shape:', delta.shape)
            if isinstance(layer, Sequentiel):
                delta = layer.backward_update_gradient(input_i, delta)
            else:
                layer.backward_update_gradient(input_i, delta)
                delta = layer.backward_delta(input_i, delta)
            if self._debug:
                print('delta l shape:', delta.shape)
                print('input shape:', input_i.shape)


        return delta 

    def update_parameters(self, gradient_step=1e-3):
        for layer in self.layers:
            layer.update_parameters(gradient_step)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

class Optim:
    def __init__(self, net, loss, eps):
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self, batch_x, batch_y):
        self.net.zero_grad()
        y_hat = self.net.forward(batch_x)
        loss = self.loss.forward(batch_y, y_hat)
        delta = self.loss.backward(batch_y, y_hat)
        self.net.backward_update_gradient(batch_x, delta)
        self.net.update_parameters(self.eps)
        return loss.mean() 

class Optim_old:
    def __init__(self, net, loss, eps):
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self, batch_x, batch_y):
        self.net.zero_grad()
        self.net.backward(batch_x, batch_y, self.loss)
        self.net.update_parameters(self.eps)


class Softmax(Module):
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward_delta(self, x, delta):

        return delta
    
    def backward_update_gradient(self, X, delta):
        pass

    def update_parameters(self, lr):
        pass

    def zero_grad(self):
        pass

class LogSoftmax(Module):
    def forward(self, x):
        self.x = x
        logsumexp = np.log(np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True))
        return x - np.max(x, axis=1, keepdims=True) - logsumexp

    def backward_delta(self, x, delta):
        exp_x = np.exp(self.x - np.max(self.x, axis=1, keepdims=True))
        softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return delta - np.sum(delta * softmax, axis=1, keepdims=True)
    
    def backward_update_gradient(self, X, delta):
        pass

    def update_parameters(self, lr):
        pass

    def zero_grad(self):
        pass


class WordEncoder_sep(Module):
    def __init__(self, word_size, alphabet_size, letter_embedding_dim, word_embedding_dim, letter_layers=[25], word_layers=[50], init=None):
        super().__init__()
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.letter_embedding_dim = letter_embedding_dim
        self.word_embedding_dim = word_embedding_dim

        layer = [Linear(alphabet_size, letter_layers[0], init), TanH()]
        for i in range(1, len(letter_layers)):
            layer.append(Linear(letter_layers[i-1], letter_layers[i], init))
            layer.append(TanH())
        layer.append(Linear(letter_layers[-1], letter_embedding_dim, init))
        layer.append(TanH())
        self.letters_encoder = [Sequentiel(*layer) for _ in range(word_size)]


        layer = [Linear(word_size * letter_embedding_dim, word_layers[0], init), TanH()]
        for i in range(1, len(word_layers)):
            layer.append(Linear(word_layers[i-1], word_layers[i], init))
            layer.append(TanH())
        layer.append(Linear(word_layers[-1], word_embedding_dim, init))
        layer.append(TanH())
        self.word_encoder = Sequentiel(*layer)

        layer = [Linear(word_embedding_dim, word_layers[-1], init), TanH()]
        for i in range(len(word_layers)-1, 0, -1):
            layer.append(Linear(word_layers[i], word_layers[i-1], init))
            layer.append(TanH())
        layer.append(Linear(word_layers[0], word_size * letter_embedding_dim, init))
        layer.append(TanH())
        self.word_decoder = Sequentiel(*layer)
        # self.word_encoder = Sequentiel(Linear(word_size * letter_embedding_dim, word_layer_size, init),
        #                                TanH(),
        #                                Linear(word_layer_size, word_embedding_dim, init))
        # self.word_decoder = Sequentiel(Linear(word_embedding_dim, word_layer_size, init),
        #                                 TanH(),
        #                                 Linear(word_layer_size, word_size * letter_embedding_dim, init))

        layer = [Linear(letter_embedding_dim, letter_layers[-1], init), TanH()]
        for i in range(len(letter_layers)-1, 0, -1):
            layer.append(Linear(letter_layers[i], letter_layers[i-1], init))
            layer.append(TanH())
        layer.append(Linear(letter_layers[0], alphabet_size, init))
        layer.append(Sigmoide())
        self.letters_decoder = [Sequentiel(*layer) for _ in range(word_size)]
            # self.letters_decoder.append(Sequentiel(
            #     Linear(letter_embedding_dim, letter_layer_size),
            #     TanH(),
            #     Linear(letter_layer_size, alphabet_size),
            #     Sigmoide()
            # ))
    
    def forward(self, x):
        return self.decode(self.encode(x))
        
    def backward_delta(self, x, delta):
        batch_size = x.shape[0]
        decoded_letters_delta = []
        # On récupère les deltas de chaque couche

        for i in range(self.word_size):
            delta_letter_i = delta[:, i, :]  # gradient sur la sortie de lettres_decoder[i]
            d_letter_emb = self.letters_decoder[i].backward_delta(None, delta_letter_i)  # (B, E)
            decoded_letters_delta.append(d_letter_emb)

        delta_word_decoder_out = np.concatenate(decoded_letters_delta, axis=1)  # (B, W*E)

        delta_word_emb = self.word_decoder.backward_delta(None, delta_word_decoder_out)  # (B, D)

        delta_word_input = self.word_encoder.backward_delta(None, delta_word_emb)  # (B, W*E)

        delta_letters_enc = np.split(delta_word_input, self.word_size, axis=1)  # liste de (B, E)
        delta_input = []
        for i in range(self.word_size):
            d_input_i = self.letters_encoder[i].backward_delta(x[:, i, :], delta_letters_enc[i])  # (B, A)
            delta_input.append(d_input_i)

        # Stack pour remettre en forme (B, W, A)
        return np.stack(delta_input, axis=1)
    
    def backward_update_gradient(self, x, delta):
        batch_size = x.shape[0]

        # On récupère les outputs de chaque couche

        encoded_letters = []
        for i in range(self.word_size):
            letter_input = x[:, i, :]  # (B, A)
            encoded = self.letters_encoder[i].forward(letter_input)  # (B, E)
            encoded_letters.append(encoded)

        word_input = np.concatenate(encoded_letters, axis=1)  # (B, W*E)

        word_embedding = self.word_encoder.forward(word_input)  # (B, D)

        decoded_word = self.word_decoder.forward(word_embedding)  # (B, W*E)
        decoded_letters = np.split(decoded_word, self.word_size, axis=1)  # liste de (B, E)

        decoded_letters_delta = []
        for i in range(self.word_size):
            d_letter_emb = self.letters_decoder[i].backward_update_gradient(decoded_letters[i], delta[:, i, :])
            # d_letter_emb = self.letters_decoder[i].backward_delta(decoded_letters[i], delta[:, i, :])
            decoded_letters_delta.append(d_letter_emb)

        delta_word_decoder_out = np.concatenate(decoded_letters_delta, axis=1)  # (B, W*E)
        delta_word_emb = self.word_decoder.backward_update_gradient(word_embedding, delta_word_decoder_out)
        # delta_word_emb = self.word_decoder.backward_delta(word_embedding, delta_word_decoder_out)

        delta_word_input = self.word_encoder.backward_update_gradient(word_input, delta_word_emb)
        # delta_word_input = self.word_encoder.backward_delta(word_input, delta_word_emb)

        delta_letters_enc = np.split(delta_word_input, self.word_size, axis=1)
        for i in range(self.word_size):
            self.letters_encoder[i].backward_update_gradient(x[:, i, :], delta_letters_enc[i])

    def encode(self, x):
        encoded_letters = []
        for i in range(self.word_size):
            letter_input = x[:, i, :]
            letter_encoded = self.letters_encoder[i].forward(letter_input)
            encoded_letters.append(letter_encoded)

        word_input = np.concatenate(encoded_letters, axis=1)
        word_embedding = self.word_encoder.forward(word_input)

        return word_embedding
    
    def decode(self, word_embedding):
        decoded_word = self.word_decoder.forward(word_embedding)
        decoded_letters = np.split(decoded_word, self.word_size, axis=1)

        output_letters = []
        for i in range(self.word_size):
            out_letter = self.letters_decoder[i].forward(decoded_letters[i])
            output_letters.append(out_letter)

        y_hat = np.stack(output_letters, axis=1)
        return y_hat
    
    def update_parameters(self, lr):
        for i in range(self.word_size):
            self.letters_encoder[i].update_parameters(lr)
            self.letters_decoder[i].update_parameters(lr)
        self.word_encoder.update_parameters(lr)
        self.word_decoder.update_parameters(lr)

class WordEncoder(Module):
    def __init__(self, word_size, alphabet_size, letter_embedding_dim, word_embedding_dim, letter_layers=[25], word_layers=[50], init=None):
        super().__init__()
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.letter_embedding_dim = letter_embedding_dim
        self.word_embedding_dim = word_embedding_dim

        layer = [Linear(alphabet_size, letter_layers[0], init), TanH()]
        for i in range(1, len(letter_layers)):
            layer.append(Linear(letter_layers[i-1], letter_layers[i], init))
            layer.append(TanH())
        layer.append(Linear(letter_layers[-1], letter_embedding_dim, init))
        layer.append(TanH())
        self.letter_encoder = Sequentiel(*layer)


        layer = [Linear(word_size * letter_embedding_dim, word_layers[0], init), TanH()]
        for i in range(1, len(word_layers)):
            layer.append(Linear(word_layers[i-1], word_layers[i], init))
            layer.append(TanH())
        layer.append(Linear(word_layers[-1], word_embedding_dim, init))
        layer.append(TanH())
        self.word_encoder = Sequentiel(*layer)

        layer = [Linear(word_embedding_dim, word_layers[-1], init), TanH()]
        for i in range(len(word_layers)-1, 0, -1):
            layer.append(Linear(word_layers[i], word_layers[i-1], init))
            layer.append(TanH())
        layer.append(Linear(word_layers[0], word_size * letter_embedding_dim, init))
        layer.append(TanH())
        self.word_decoder = Sequentiel(*layer)

        layer = [Linear(letter_embedding_dim, letter_layers[-1], init), TanH()]
        for i in range(len(letter_layers)-1, 0, -1):
            layer.append(Linear(letter_layers[i], letter_layers[i-1], init))
            layer.append(TanH())
        layer.append(Linear(letter_layers[0], alphabet_size, init))
        layer.append(Sigmoide())
        self.letter_decoder = Sequentiel(*layer)
    
    def encode(self, x):
        # x shape : (B, W, A)
        batch_size = x.shape[0]
        encoded_letters = []

        for i in range(self.word_size):
            letter_input = x[:, i, :]  # (B, A)
            letter_encoded = self.letter_encoder.forward(letter_input)  # (B, E)
            encoded_letters.append(letter_encoded)

        word_input = np.concatenate(encoded_letters, axis=1)  # (B, W*E)
        word_embedding = self.word_encoder.forward(word_input)  # (B, D)
        return word_embedding

    def decode(self, word_embedding):
        decoded_word = self.word_decoder.forward(word_embedding)  # (B, W*E)
        decoded_letters = np.split(decoded_word, self.word_size, axis=1)  # W x (B, E)

        output_letters = []
        for i in range(self.word_size):
            out_letter = self.letter_decoder.forward(decoded_letters[i])  # (B, A)
            output_letters.append(out_letter)

        y_hat = np.stack(output_letters, axis=1)  # (B, W, A)
        return y_hat

    def forward(self, x):
        return self.decode(self.encode(x))

    def backward_delta(self, x, delta):
        # x: (B, W, A), delta: (B, W, A)
        decoded_letters_delta = []
        encoded_letters = []

        # Re-encode les lettres
        for i in range(self.word_size):
            letter_input = x[:, i, :]
            encoded = self.letter_encoder.forward(letter_input)
            encoded_letters.append(encoded)

        word_input = np.concatenate(encoded_letters, axis=1)
        word_embedding = self.word_encoder.forward(word_input)
        decoded_word = self.word_decoder.forward(word_embedding)
        decoded_letters = np.split(decoded_word, self.word_size, axis=1)

        # backward delta des sorties lettres
        for i in range(self.word_size):
            d_letter_emb = self.letter_decoder.backward_delta(decoded_letters[i], delta[:, i, :])
            decoded_letters_delta.append(d_letter_emb)

        delta_word_decoder_out = np.concatenate(decoded_letters_delta, axis=1)
        delta_word_emb = self.word_decoder.backward_delta(word_embedding, delta_word_decoder_out)
        delta_word_input = self.word_encoder.backward_delta(word_input, delta_word_emb)
        delta_letters_enc = np.split(delta_word_input, self.word_size, axis=1)

        delta_input = []
        for i in range(self.word_size):
            d_input_i = self.letter_encoder.backward_delta(x[:, i, :], delta_letters_enc[i])
            delta_input.append(d_input_i)

        return np.stack(delta_input, axis=1)

    def backward_update_gradient(self, x, delta):
        encoded_letters = []

        for i in range(self.word_size):
            letter_input = x[:, i, :]
            letter_encoded = self.letter_encoder.forward(letter_input)
            encoded_letters.append(letter_encoded)

        word_input = np.concatenate(encoded_letters, axis=1)
        word_embedding = self.word_encoder.forward(word_input)
        decoded_word = self.word_decoder.forward(word_embedding)
        decoded_letters = np.split(decoded_word, self.word_size, axis=1)

        decoded_letters_delta = []
        for i in range(self.word_size):
            d_letter_emb = self.letter_decoder.backward_update_gradient(decoded_letters[i], delta[:, i, :])
            # d_letter_emb = self.letter_decoder.backward_delta(decoded_letters[i], delta[:, i, :])
            decoded_letters_delta.append(d_letter_emb)

        delta_word_decoder_out = np.concatenate(decoded_letters_delta, axis=1)
        delta_word_emb = self.word_decoder.backward_update_gradient(word_embedding, delta_word_decoder_out)
        # delta_word_emb = self.word_decoder.backward_delta(word_embedding, delta_word_decoder_out)

        delta_word_input = self.word_encoder.backward_update_gradient(word_input, delta_word_emb)
        # delta_word_input = self.word_encoder.backward_delta(word_input, delta_word_emb)

        delta_letters_enc = np.split(delta_word_input, self.word_size, axis=1)
        for i in range(self.word_size):
            self.letter_encoder.backward_update_gradient(x[:, i, :], delta_letters_enc[i])
    
    def update_parameters(self, lr):
        self.letter_encoder.update_parameters(lr)
        self.letter_decoder.update_parameters(lr)
        self.word_encoder.update_parameters(lr)
        self.word_decoder.update_parameters(lr)


