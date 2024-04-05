function _sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// Tensor class
class Tensor {
    constructor(data, shape) {
        this.data = data;
        this.shape = shape;
        this.stride = Array(this.shape.length);
        let acc = 1
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
            this.stride[i] = 0;
        }
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
}

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
        this.y.backward(loss.mul(this.x.val.apply_binary_op(this.y.val, this.helper)));
    };
}

class DotBackward {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
    call(loss) {
        this.x.backward(loss.dot(this.y.transpose(0, 1)));
        this.y.backward(this.x.transpose(0, 1).dot(loss));
    };
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

// Variable
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
    backward(loss) {
        if (this.backward_hook) {
            this.backward_hook.call(loss);
        } else {
            this.grad = this.grad.add(loss);
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
}

function sigmoid(v) {
    let new_val = v.val.apply_unitary_op(_sigmoid);
    let o = new Variable(new_val, new NopBackward());
    o.backward_hook = new SigmoidBackward(v, o);
    return o;
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
        return x.dot(this.W).add(b)
    }
}

class Sigmoid {
    constructor() {

    }

    forward(x) {
        return sigmoid(x)
    }
}

let x = new Tensor([1, 2, 3, 4], [2, 2]);
let y = x.dot(x.transpose(0, 1))

console.log(y)