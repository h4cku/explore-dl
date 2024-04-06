// Activation Functions 
function _sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function _tanh(x) {
    return (1 - Math.exp(-x)) / (1 + Math.exp(-x));
}

function _relu(x) {
    return x > 0 ? x : 0;
}

function _gelu(x) {
    return x * _sigmoid(1.702 * x);
}

function _silu(x) {
    return x * _sigmoid(x);
}

// Tensor class
class Tensor {
    constructor(data, shape) {
        this.data = data;
        this.shape = shape;
        this.stride = Array(this.shape.length);
        this.broadcast_idx = -1;
        let acc = 1;
        for (let i = this.shape.length - 1; i >= 0; i--) {
            this.stride[i] = acc;
            acc *= this.shape[i];
        }
    }

    match_dimension(m) {
        if (this.shape.length != m.shape.length) {
            return false;
        }
        for (let i = 0; i < this.shape.length; i++) {
            if (m.stride[i] != 0 && this.shape[i] != m.shape[i]) {
                return false;
            }
        }
        return true;
    }

    broadcast(i) {
        if (this.shape[i] == 1) {
            this.broadcast_idx = i;
            this.stride[i] = 0;
        }
    }

    get_broadcast_idx() {
        return this.broadcast_idx;
    }

    reshape(new_shape) {
        this.shape = new_shape;
        this.stride = Array(this.shape.length);
        let acc = 1
        for (let i = this.shape.length - 1; i >= 0; i--) {
            this.stride[i] = acc;
            acc *= this.shape[i];
        }
    }

    set_stride(new_stride) {
        this.stride = new_stride;
    }

    random() {
        for (let i = 0; i < this.data.length; i++) {
            this.data[i] = Math.random();
        }
    }

    zeros() {
        for (let i = 0; i < this.data.length; i++) {
            this.data[i] = 0;
        }
    }

    ones() {
        for (let i = 0; i < this.data.length; i++) {
            this.data[i] = 1;
        }
    }

    at(idx) {
        let s = 0;
        for (let i = 0; i < idx.length; i++) {
            s += idx[i] * this.stride[i];
        }
        return this.data[s];
    }

    set(idx, val) {
        let s = 0;
        for (let i = 0; i < idx.length; i++) {
            s += idx[i] * this.stride[i];
        }
        this.data[s] = val;
    }

    get_idx(i) {
        let idx = [];
        for (let j = 0; j < this.stride.length; j++) {
            let t_ = Math.floor(i / this.stride[j]);
            i = i - t_ * this.stride[j];
            idx.push(t_)
        }
        return idx;
    }

    get_pos(idx) {
        let s = 0;
        for (let i = 0; i < idx.length; i++) {
            s += idx[i] * this.stride[i];
        }
        return s;
    }

    get_dim() {
        let acc = 1;
        for (let i = 0; i < this.shape.length; i++) {
            acc *= this.shape[i];
        }
        return acc;
    }

    apply_unitary_op(f) {
        let new_data = []
        for (let i = 0; i < this.data.length; i++) {
            new_data.push(f(this.data[i]));
        }
        return new Tensor(new_data, [... this.shape]);
    }

    apply_binary_op(m, f) {
        if (!this.match_dimension(m)) {
            return null;
        }
        let new_data = []
        for (let i = 0; i < this.data.length; i++) {
            let idx = this.get_idx(i);
            new_data.push(f(this.data[i], m.at(idx)));
        }
        return new Tensor(new_data, [... this.shape]);
    }

    sum(axis) {
        let new_shape = [...this.shape];
        new_shape[axis] = 1;
        let o = new Tensor(Array(this.get_dim() / this.shape[axis]), new_shape);
        o.zeros();
        for (let i = 0; i < this.data.length; i++) {
            let idx = this.get_idx(i);
            idx[axis] = 0;
            o.data[o.get_pos(idx)] += this.data[i];
        }
        return o;
    }

    mean(axis) {
        let o = this.sum(axis);
        o = o.times(1 / this.shape[axis]);
        return o
    }

    add(m) {
        if (!this.match_dimension(m)) {
            return null;
        }
        let new_data = []
        for (let i = 0; i < this.data.length; i++) {
            let idx = this.get_idx(i);
            new_data.push(this.data[i] + m.at(idx));
        }
        return new Tensor(new_data, [... this.shape]);
    }

    sub(m) {
        if (!this.match_dimension(m)) {
            return null;
        }
        let new_data = []
        for (let i = 0; i < this.data.length; i++) {
            let idx = this.get_idx(i);
            new_data.push(this.data[i] - m.at(idx));
        }
        return new Tensor(new_data, [... this.shape]);
    }

    mul(m) {
        if (!this.match_dimension(m)) {
            return null;
        }
        let new_data = []
        for (let i = 0; i < this.data.length; i++) {
            let idx = this.get_idx(i);
            new_data.push(this.data[i] * m.at(idx));
        }
        return new Tensor(new_data, [... this.shape]);
    }

    div(m) {
        if (!this.match_dimension(m)) {
            return null;
        }
        let new_data = []
        for (let i = 0; i < this.data.length; i++) {
            let idx = this.get_idx(i);
            new_data.push(this.data[i] / m.at(idx));
        }
        return new Tensor(new_data, [... this.shape]);
    }

    times(s) {
        let new_data = []
        for (let i = 0; i < this.data.length; i++) {
            new_data.push(this.data[i] * s);
        }
        return new Tensor(new_data, [... this.shape]);
    }

    dot(m) {
        // TO IMPROVE
        let o = new Tensor(Array(this.shape[0] * m.shape[1]), [this.shape[0], m.shape[1]]);
        for (let i = 0; i < this.shape[0]; i++) {
            for (let j = 0; j < m.shape[1]; j++) {
                let s = 0;
                for (let k = 0; k < this.shape[1]; k++) {
                    s += this.at([i, k]) * m.at([k, j]);
                }
                o.set([i, j], s);
            }
        }
        return o;
    }

    transpose(i, j) {
        let o = new Tensor(this.data, [...this.shape]);

        let x_ = o.stride[i];
        o.stride[i] = o.stride[j];
        o.stride[j] = x_;

        x_ = o.shape[i];
        o.shape[i] = o.shape[j];
        o.shape[j] = x_;

        return o;
    }
    pow(e) {
        let new_data = []
        for (let i = 0; i < this.data.length; i++) {
            new_data.push(Math.pow(this.data[i], e));
        }
        return new Tensor(new_data, [... this.shape]);
    }
}

// Backward Classes
class NopBackward {
    constructor() {

    }
    call(loss) {

    }
}

class AddBackward {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
    call(loss) {
        this.x.backward(loss);
        if (this.y.val.get_broadcast_idx() >= 0)
            this.y.backward(loss.sum(this.y.val.get_broadcast_idx()))
        else
            this.y.backward(loss);
    }
}

class SubBackward {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
    call(loss) {
        this.x.backward(loss);
        if (this.y.val.get_broadcast_idx() >= 0)
            this.y.backward(loss.sum(this.y.val.get_broadcast_idx()).times(-1))
        else
            this.y.backward(loss.times(-1));
    }
}

class MulBackward {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
    call(loss) {
        this.x.backward(loss.mul(y.val));
        if (this.y.val.get_broadcast_idx() >= 0)
            this.y.backward(loss.mul(x.val).sum(this.y.val.get_broadcast_idx()))
        else
            this.y.backward(loss.mul(x.val));
    }
}

class DivBackward {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
    helper(a, b) {
        return -a / (b ** 2);
    }
    call(loss) {
        this.x.backward(loss.div(this.y.val));
        if (this.y.val.get_broadcast_idx() >= 0)
            this.y.backward(loss.mul(this.x.val.apply_binary_op(this.y.val, this.helper)).sum(this.y.val.get_broadcast_idx()))
        else
            this.y.backward(loss.mul(this.x.val.apply_binary_op(this.y.val, this.helper)));
    };
}

class DotBackward {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
    call(loss) {
        this.x.backward(loss.dot(this.y.val.transpose(0, 1)));
        this.y.backward(this.x.val.transpose(0, 1).dot(loss));
    };
}

class PowBackward {
    constructor(x, o, e) {
        this.x = x;
        this.o = o;
        this.e = e;
    }
    call(loss) {
        this.x.backward(loss.mul(this.o.val.div(this.x.val).times(this.e)))
    }
}

class MeanBackward {
    constructor(x, axis) {
        this.x = x;
        this.axis = axis;
    }
    call(loss) {
        let t_ = new Tensor(Array(this.x.val.get_dim()), this.x.val.shape);
        t_.ones();
        loss.broadcast(this.axis);
        t_ = t_.mul(loss)
        this.x.backward(t_.times(1 / this.x.val.shape[this.axis]));
    }
}

class SigmoidBackward {
    constructor(x, o) {
        this.x = x;
        this.o = o;
    }
    helper(x) {
        return x * (1 - x)
    }
    call(loss) {
        this.x.backward(loss.mul(this.o.val.apply_unitary_op(this.helper)))
    }
}

class TanhBackward {
    constructor(x, o) {
        this.x = x;
        this.o = o;
    }
    helper(x) {
        return (1 - x ** 2) / 2
    }
    call(loss) {
        this.x.backward(loss.mul(this.o.val.apply_unitary_op(this.helper)))
    }
}

// Variable class 
// This a wrapper for tensor objects to be able to perform autodifferntiation
class Variable {
    constructor(t, backward_hook) {
        this.val = t;
        if (this.backward_hook) {
            this.grad = null;
        } else {
            this.grad = new Tensor(Array(t.data.length), [...t.shape]);
        }
        this.grad.zeros();
        this.backward_hook = backward_hook; // if this is null it means it is a leaf of the graph
    }
    broadcast(i) {
        this.val.broadcast(i);
    }
    backward(loss) {
        if (this.backward_hook) {
            this.backward_hook.call(loss);
        } else {
            this.grad = this.grad.add(loss);
        }
    }
    zero_grad() {
        if (this.grad) {
            this.grad.zeros()
        }
    }
    add(v) {
        let new_val = this.val.add(v.val);
        return new Variable(new_val, new AddBackward(this, v));
    }
    sub(v) {
        let new_val = this.val.sub(v.val);
        return new Variable(new_val, new SubBackward(this, v));
    }
    mul(v) {
        let new_val = this.val.mul(v.val);
        return new Variable(new_val, new MulBackward(this, v));
    }
    div(v) {
        let new_val = this.val.div(v.val);
        return new Variable(new_val, new DivBackward(this, v));
    }
    dot(v) {
        let new_val = this.val.dot(v.val);
        return new Variable(new_val, new DotBackward(this, v));
    }
    mean(axis) {
        let new_val = this.val.mean(axis);
        return new Variable(new_val, new MeanBackward(this, axis));
    }
    pow(e) {
        let new_val = this.val.pow(e);
        let o = new Variable(new_val, new NopBackward());
        o.backward_hook = new PowBackward(this, o, e);
        return o;
    }
    static sigmoid(v) {
        let new_val = v.val.apply_unitary_op(_sigmoid);
        let o = new Variable(new_val, new NopBackward());
        o.backward_hook = new SigmoidBackward(v, o);
        return o;
    }
    static tanh(v) {
        let new_val = v.val.apply_unitary_op(_tanh);
        let o = new Variable(new_val, new NopBackward());
        o.backward_hook = new TanhBackward(v, o);
        return o;
    }
}

// Loss functions
class Loss {
    static mse_loss(o, t) {
        return o.sub(t).mean(0).mean(1)
    }

    static cross_entropy_loss() {

    }
}

// Optimizers

class SGD {
    constructor(params, hparams) {
        this.params = params;
        this.lr = hparams.lr;
    }

    zero_grad() {
        for (let i = 0; i < this.params.length; i++) {
            this.params[i].zero_grad()
        }
    }

    step() {
        for (let i = 0; i < this.params.length; i++) {
            this.params[i].data -= this.lr * this.params[i].grad
        }
    }
}

// Nerual Network Layers
class Linear {
    constructor(n_input, n_output) {
        let W_ = new Tensor(Array(n_input * n_output), [n_input, n_output]);
        let b_ = new Tensor(Array(n_output), [1, n_output]);
        W_.random()
        b_.random()
        this.W = new Variable(W_, null);
        this.b = new Variable(b_, null)
    }

    forward(x) {
        return x.dot(this.W).add(this.b)
    }
}

class Sigmoid {
    constructor() {

    }

    forward(x) {
        return Variable.sigmoid(x)
    }
}

class Tanh {
    constructor() {

    }

    forward(x) {
        return Variable.sigmoid(x)
    }
}

class Sequential {
    constructor(layers) {
        this.layers = layers;
    }

    forward(x) {
        let o = x;
        for (let i = 0; i < this.layers.length; i++) {
            o = this.layers[i].forward(o)
        }
        return o;
    }
}

let model = new Sequential([
    new Linear(2, 5),
    new Sigmoid(),
    new Linear(5, 1)
])

let x = new Variable(new Tensor([2, 3], [1, 2]), new NopBackward())
console.log(model.forward(x))
