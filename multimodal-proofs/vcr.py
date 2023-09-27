# https://rowanzellers.com/merlotreserve/
# https://github.com/rowanz/merlot_reserve

from mreserve.modeling import MerlotReserve 
import flax.linen as nn  
import jax  
import jx.numpy as jnp

jax.config.update('jax_log_compiles', True)
is_on_gpu = any([x.platform == 'gpu' for x in jax.local_devices()])
print('JAX process: {} / {}. Local devices {}. Using {}'.format(
    jax.process_index(), jax.process_count(), jax.local_devices(), 'GPU' if is_on_gpu else 'TPU'), flush=True)


class MerlotReserveVCR(MerlotReserve):
    def setup(self):
        super().setup()
        self.proj = nn.Dense(features=1, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=0.02), name='proj',
                             use_bias=False)

    def __call__(self, batch):

        batch_size, two_, num_ans_per, token_length = batch['answers'].shape
        answers2d = batch['answers'].reshape(batch_size * 2 * num_ans_per, token_length)

        imgs_enc = self.vision_encoder(batch['image'])['seq_attnpool'].repeat(2 * num_ans_per, axis=0)

        mm_inputs = self.prepare_multimodal_inputs(
            tokens=answers2d,
            token_segment_idx=jnp.zeros([batch_size * 2 * num_ans_per, token_length], dtype=jnp.int32),
            vision_input=imgs_enc,
        )
        joint_encoding = self.joint_transformer(**mm_inputs)['seq']
        joint_encoding = joint_encoding[:, :token_length].reshape(batch_size * 2 * num_ans_per, token_length, self.hidden_size)

        # Pool from the right tokens
        pool_idx = jnp.argmax((answers2d == MASK).astype(jnp.float32), 1)
        pooled_h = joint_encoding[jnp.arange(batch_size * 2 * num_ans_per), pool_idx]

        logits = self.proj(pooled_h).reshape([batch_size, 2, num_ans_per])
        return logits


def load_model():
    # ../../pretrain/configs/base.yaml
    model = MerlotReserveVCR.from_config(config)
    param = load_checkpoint(ckpt)['params']

def preprocess_data():
    feature_dict = {
            'image': bytes_feature(pil_image_to_jpgstring(ex['image'])),
            'image_fliplr': bytes_feature(pil_image_to_jpgstring(ex.get('image_fliplr', ex['image']))),
            'id': bytes_feature(ex['id'].encode('utf-8')),
        }

    for prefix in ['qa', 'qar']:
        query_enc = encoder.encode(ex[f'{prefix}_query']).ids
        feature_dict[f'{prefix}_query'] = int64_list_feature(query_enc)
        for i, choice_i in enumerate(encoder.encode_batch(ex[f'{prefix}_choices'])):
            feature_dict[f'{prefix}_choice_{i}'] = int64_list_feature(choice_i.ids)
            max_len = max(len(choice_i.ids) + len(query_enc), max_len)
        feature_dict[f'{prefix}_label'] = int64_feature(ex[f'{prefix}_label'])
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))