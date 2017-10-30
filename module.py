from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def discriminatorA(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, name='da_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='da_h1_conv'), 'da_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='da_h2_conv'), 'da_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='da_h3_conv'), 'da_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, s=1, name='da_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4

def discriminatorB(image, options, reuse=False, name='discriminatorB'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        #input 5 5 2
        h0 = lrelu(conv2d(image, options.df_dim,ks=3,s=1,name='db_h0_conv'))
        # h0 is (5 5 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2,ks=2, s=1,name='db_h1_conv'), 'db_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, ks=2,s=2,name='db_h2_conv'), 'db_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = conv2d(h2, 1, s=1, name='db_h3_pred')
        # h4 is (5 x 5 x 1)
        return h3

def generator_a2b(image, options, reuse=False, name="generatorB"):
    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'gabr_c1'), name+'gabr_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'gabr_c2'), name+'gabr_bn2')
            return y + x

        # input (256, 256, 3)
        e1 = instance_norm(conv2d(image, options.gf_dim, name='gab_e1_conv'))
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_norm(conv2d(lrelu(e1), options.gf_dim*2, name='gab_e2_conv'), 'gab_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = instance_norm(conv2d(lrelu(e2), options.gf_dim*4, name='gab_e3_conv'), 'gab_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        
        r1 = residule_block(e3, options.gf_dim*4, name='gab_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='gab_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='gab_r3')

        e4 = instance_norm(conv2d(lrelu(r3), options.gf_dim*8, name='gab_e4_conv'), 'gab_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = instance_norm(conv2d(lrelu(e4), options.gf_dim*8, name='gab_e5_conv'), 'gab_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = instance_norm(conv2d(lrelu(e5), options.gf_dim*4, name='gab_e6_conv'), 'gab_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        r4 = residule_block(e6, options.gf_dim*4, name='gab_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='gab_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='gab_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='gab_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='gab_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='gab_r9')
        # (4,4, dim*4)

        d1 = deconv2d(r9, options.gf_dim*2, ks=2, s=1, padding='VALID', name='gab_d1')
        d1 = tf.nn.relu(instance_norm(d1, 'gab_d1_bn'))
        # (5 x 5 x 64*2)
        d2 = conv2d(d1, options.gf_dim, ks=2, s=1, name='gab_d2')
        d2 = tf.nn.relu(instance_norm(d2, 'gab_d2_bn'))
        # (5 x 5 x 64)
        d3 = conv2d(d2, 2, ks=2,s=1, name='gab_pred')
        # (5 x 5 x 2)
        return d3

def generator_b2a(image, options, reuse=False, name="generatorA"):
    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'gabr_c1'), name+'gabr_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'gabr_c2'), name+'gabr_bn2')
            return y + x

        #input (5,5,2)
        e1 = conv2d(image, options.gf_dim, ks=2, s=1, name='gba_e1')
        # (5,5,64)
        e2 = instance_norm(conv2d(lrelu(e1),options.gf_dim*2,ks=2,s=1,padding='VALID',name='gba_e2'))
        # (4, 4, gf_dim*2)
        e3 = deconv2d(tf.nn.relu(e2), options.gf_dim*4, name='gba_e3')
        # (8, 8, gf_dim*4)
        
        r1 = residule_block(e3, options.gf_dim*4, name='gba_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='gba_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='gba_r3')

        e4 = deconv2d(tf.nn.relu(r3), options.gf_dim*8,name='gba_e4')
        # (16 16 gf_dim*8)
        e5 = deconv2d(tf.nn.relu(e4), options.gf_dim*8,name='gba_e5')
        # 32 32 
        e6 = deconv2d(tf.nn.relu(e5), options.gf_dim*4,name='gba_e6')
        # 64 64
        r4 = residule_block(e6, options.gf_dim*4, name='gba_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='gba_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='gba_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='gba_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='gba_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='gba_r9')

        d1 = deconv2d(r9, options.gf_dim*4, name='gba_d1')
        d1 = tf.nn.relu(instance_norm(d1, 'gba_d1_bn'))
        # (128 128)
        d2 = deconv2d(d1, options.gf_dim*2,  name='gba_d2')
        d2 = tf.nn.relu(instance_norm(d2, 'gba_d2_bn'))
        # (256 256)
        d3 = conv2d(d2, 3, ks=3,s=1, name='gba_pred')
        # (5 x 5 x 2)
        return tf.nn.tanh(d3)
        


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
