from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
import csv
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow

from module import *
from utils import *


class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.out_size = args.out_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir

        self.discriminatorA = discriminatorA
        self.discriminatorB = discriminatorB
        self.generator_a2b = generator_a2b
        self.generator_b2a = generator_b2a
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)

    def _build_model(self):

        self.real_A = tf.placeholder(tf.float32,[None, self.image_size, self.image_size, self.input_c_dim],name='real_A_images')
        self.real_B = tf.placeholder(tf.float32,[None, self.out_size, self.out_size, self.output_c_dim],name='real_B_joints')

        self.fake_B = self.generator_a2b(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator_b2a(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A = self.generator_b2a(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator_a2b(self.fake_A, self.options, True, name="generatorA2B")

        self.DB_fake = self.discriminatorB(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminatorA(self.fake_A, self.options, reuse=False, name="discriminatorA")
        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.out_size, self.out_size,
                                             self.output_c_dim], name='fake_B_sample')
        self.DB_real = self.discriminatorB(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminatorA(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminatorB(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminatorA(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss

        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )

        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.out_size, self.out_size,
                                      self.output_c_dim], name='test_B')
        self.testB = self.generator_a2b(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator_b2a(self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        B = np.loadtxt('./datasets/{}/joints2D_train.csv'.format(self.dataset_dir + '/trainB'),delimiter=',',dtype=np.float32)
        for epoch in range(args.epoch):
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = B[:]
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

            for idx in range(0, batch_idxs):
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size], 
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_a = []
                batch_b = []
                for batch_file in batch_files:
                    a, b = load_train_data(batch_file, args.load_size, args.fine_size)
                    batch_a.append(a)
                    batch_b.append(b)
                batch_a = np.array(batch_a).astype(np.float32)
                batch_b = np.array(batch_b).astype(np.float32)

                # Update G network and record fake outputs
                fake_A, fake_B, _, summary_str = self.sess.run(
                    [self.fake_A, self.fake_B, self.g_optim, self.g_sum],
                    feed_dict={self.real_A: batch_a, self.real_B: batch_b, self.lr: lr})
                self.writer.add_summary(summary_str, counter)
                [fake_A, fake_B] = self.pool([fake_A, fake_B])

                # Update D network
                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],
                    feed_dict={self.real_A: batch_a,
                               self.real_B: batch_b,
                               self.fake_A_sample: fake_A,
                               self.fake_B_sample: fake_B,
                               self.lr: lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (
                    epoch, idx, batch_idxs, time.time() - start_time)))

                if np.mod(counter, args.save_freq) == 2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        status = 0
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            status = 1
            sample_files = np.loadtxt('./datasets/{}/joints2D_test.csv'.format(self.dataset_dir + '/testB'),delimiter=',',dtype=np.float32)
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else (
            self.testA, self.test_B)
        colors = rainbow(np.linspace(0,1,24))
        cs = [colors[i] for i in range(24)]

        if status == 0: 
            fake_joints = []
            for sample_file in sample_files:
                print('Processing image: ' + sample_file)
                sample_image = [load_test_data(sample_file, args.fine_size)]
                sample_image = np.array(sample_image).astype(np.float32)
                
                fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
                fake_joints.append(fake_img.flatten())
                fake_img = fake_img.reshape(25,2).T
                image = Image.open(sample_file)
                plt.imshow(image)
                plt.axis('off')
                plt.scatter(fake_img[0], fake_img[1], color=cs)
                name = os.path.basename(sample_file)
                new_path = '{}/joints_{}.png'.format(args.test_dir, name[:-4])
                plt.savefig(new_path)
                plt.gcf().clear()
                index.write("<td>%s</td>" % name)
                index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                    '..' + os.path.sep + sample_file)))
                #index.write("<td>'%s'</td>" % (" ".join(str(elm) for elm in fake_img)))
                index.write("<td><img src='%s'></td>" % new_path)
                index.write("</tr>")
            index.close()
            file_path = os.path.join(args.test_dir, '{}.csv'.format(args.which_direction))
            with open(file_path, 'w') as outfile:
                writer = csv.writer(outfile, lineterminator='\n')
                writer.writerows(fake_joints)
        else:
            for i, sample in enumerate(sample_files):
                sample = sample.reshape(self.output_c_dim, self.out_size*self.out_size).T.reshape(self.out_size, self.out_size, self.output_c_dim)
                image_path = os.path.join(args.test_dir,
                                          '{0}_{1}.jpg'.format(args.which_direction, str(i)))
                fake_img = self.sess.run(out_var, feed_dict={in_var: sample})
                save_images(fake_img, [1, 1], image_path)
                index.write("<td>%s</td>" % str(i))
                index.write("<td>'%s'</td>" % (" ".join(str(elm) for elm in sample)))
                index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                    '..' + os.path.sep + image_path)))
                index.write("</tr>")
            index.close()


