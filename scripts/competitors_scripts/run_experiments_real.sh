#!/bin/bash

set -x

while getopts "i:l:R:r:t:v:" opt
do
    case ${opt} in
        i) INPUT_FOLDER=${OPTARG} ;;
        l) LABEL=${OPTARG} ;;
        R) V_OFFSET_RADIUS=${OPTARG} ;;
        r) V_CONV_RADIUS=${OPTARG} ;;
        t) V_THRESHOLD=${OPTARG} ;;
        v) VARLEN=${OPTARG} ;;
    esac
done


for filename in abc_0050_00500041_5aa40dcd43fa0b14df9bdcf8_010 
# abc_0050_00500082_4cb4bf14428fe3832dd7ed78_000 abc_0050_00500136_5894b99c0d2ba10fcbeff634_000 abc_0050_00500149_54930d6f7740b03347d89a56_000 abc_0050_00500166_5894bbd701b2bb0fc88a6978_007 abc_0050_00500348_fae0ecd8b3dc068d39f0d09c_000 abc_0050_00500683_511f7debb63f164003339dec_000 abc_0050_00500912_ded12ea6d13dadf860aaa4cb_000 abc_0050_00501216_5894ff84fca3da0f6b67497b_012 abc_0050_00502539_6e851dae74079e807f2dbcf4_012 abc_0050_00505574_5897be3fa417220f72333795_000 abc_0050_00506371_589844ae8460820fe9280f95_058 abc_0050_00506880_f982cf821a90f24a03bd661f_000 abc_0050_00507625_b39680dc8dc650ec01956bbe_011 abc_0050_00508845_266ea1913e2d35805fb70e9d_001 abc_0050_00509172_c683c0212890fcb20340e86c_000 abc_0051_00510038_1509fe78c7a4729189287091_000 abc_0051_00510073_951e25d2ded40f22b598f84e_000 abc_0051_00510077_3cb33817515aa6a5e8dd638d_000 abc_0051_00510118_e88bd32150a7b44caee0fd29_003 abc_0051_00510141_2b44a82f2bf7942f43634723_000 abc_0051_00510218_0539d1001509ed5da517b042_000 abc_0051_00510270_14214b7927680bf60e4a08f4_000 abc_0051_00510325_0769ac8e70aa984887cf08ec_000 abc_0051_00510333_44cac8b5175622647f4436a1_000 abc_0051_00510527_bb814ec02a7f1d6ec7ee7af4_000 abc_0051_00510545_045b506b58f91329b9fac90a_000 abc_0051_00510577_71536d4c280b04b7f4a05310_003 abc_0051_00510848_589a11a3e0d0da0f7abae8a6_003 abc_0051_00510882_589a14d663f4f81027166c63_001 abc_0051_00511007_e70e24b5c39dad8ca30798dc_000 abc_0051_00511067_09f6da0fad2ef45888848f34_000 abc_0051_00511304_d7883809a97367c134e262f6_001 abc_0051_00511357_6b62490b84f2abcfe801837e_001 abc_0051_00511469_07a2c0c1151fc7616ad0c39e_000 abc_0051_00511801_8dd40abaa1cd649cca2577c2_005 abc_0051_00511846_7b505410617e2dccd4c756dc_013 abc_0051_00512045_589a74c5e0d0da0f7abb7bad_002 abc_0051_00512087_589a74c5e0d0da0f7abb7bad_044 abc_0051_00512117_589a74c5e0d0da0f7abb7bad_074 abc_0051_00512224_589a795c7e864b0f65045295_060 abc_0051_00512677_124a782a9121432e35fb91fd_000 abc_0051_00512830_589abe4f8f313d0fc5e2666c_004 abc_0051_00512867_bb8ce171738b5deacb786b2d_008 abc_0051_00512990_42b40cd44dd76271faeeddc0_000 abc_0051_00513141_589af21e8460820fe92a2bef_000 abc_0051_00513464_4990b47baa768a15e5b6f2fa_000 abc_0051_00513656_61ae550d0c4597033dd4ab99_000 abc_0051_00513771_589b1e411529400fd28c70ee_069 abc_0051_00513811_589b21908460820fe92a52ff_000 abc_0051_00514031_1211c44d63055dd205610531_002 abc_0051_00514267_589b46a41f84c20fc8bf6bfa_001 abc_0051_00514480_6c33de245ad4c4ff41a3360f_000 abc_0051_00514494_18b4250417be46b2d6fac884_000 abc_0051_00514605_c19052d705922cd25ac6e93a_005 abc_0051_00514617_c543ec2cdc61c000ae037e76_000 abc_0051_00514733_d29e9745aa348b6c6b390749_000 abc_0051_00514737_1b29e559d47a5e68f01294db_000 abc_0051_00514821_40a8d0e2ef761deda49fe44b_000 abc_0051_00514829_d05a37c045f46b2cf41639c7_000 abc_0051_00514908_11c98fc55866ff877cfcdd46_000 abc_0051_00514932_8f59ca8e0640e2d7aeb1d9e1_008 abc_0051_00514939_2e44d41f490e21f75afc031e_000 abc_0051_00515031_e40ba45a0bd5fbab7ce6b155_000 abc_0051_00515110_43a0e83aaca82954cacdbe14_005 abc_0051_00515209_339016deab3d8d5c379667e0_000 abc_0051_00515357_211afe9d7c17daea9f65e7fa_000 abc_0051_00515491_589baf19c88d2e0f6cbb9cbb_015 abc_0051_00515501_91a84e96f5d5519cd3938dcb_000 abc_0051_00515566_589bb9e8e1c77e0ffe2fcb60_023 abc_0051_00515662_08a4c17e669921ce4fc67435_001 abc_0051_00515671_589bc577e991810f6c5530f6_006 abc_0051_00515738_6a626f99dbc0bbfd0c786022_000 abc_0051_00515778_efcde9d8e66cecd7fd2f4013_004 abc_0051_00515797_589bdec38006660f7f91d8e2_001 abc_0051_00515801_589bdec38006660f7f91d8e2_005 abc_0051_00515820_589bdec38006660f7f91d8e2_024 abc_0051_00515868_589bdec38006660f7f91d8e2_072 abc_0051_00515869_589bdec38006660f7f91d8e2_073 abc_0051_00515877_589bdec38006660f7f91d8e2_081 abc_0051_00515913_f2047825d47c9e1474ad63fa_000 abc_0051_00515956_654ec074c8d83a0b57e46110_004 abc_0051_00516126_77035ed260312e601293a18f_000 abc_0051_00516159_9277ec867147964cea5ea57d_000 abc_0051_00516167_2b4007f8c4663b6dbdf91c6b_003 abc_0051_00516184_e61c9fb26582c5967d218009_000 abc_0051_00516261_ffb04e7d1767feb7af09f774_000 abc_0051_00516341_f2762df970f5541acaf37038_000 abc_0051_00516364_1bb8f06910990563c3e12991_000 abc_0051_00516398_3d8f5ccaf54e6a12ab4a2153_000 abc_0051_00516399_3d8f5ccaf54e6a12ab4a2153_001 abc_0051_00516407_36ff287f834a1627a4b5d1c9_000 abc_0051_00516689_4581cda73ff1f27569d69fc6_002 abc_0051_00516710_b024abfa327cefcd8d8e882d_000 abc_0051_00516825_8f4c8b2da96baf82ecaecf90_003 abc_0051_00516874_589c9226d978e50f89dfb381_008 abc_0051_00516985_701b01041577737beea11fc2_003 abc_0051_00516996_76ad2d8473bb693d8a9c70db_000 abc_0051_00517328_d8d7899e536c5464c1b02340_000 abc_0051_00517412_589cc2d908d6170f75acb3e2_002 abc_0051_00517445_bd3ff6f42017893b7d43a805_000 abc_0051_00517548_f13866b5bc2af32ab038252f_000 abc_0051_00517560_6b1e65e999a752d65261fccb_000 abc_0051_00517689_6fe915d6aada5c49c7c7ed68_000 abc_0051_00517707_ab1035f9186293171f11021f_000 abc_0051_00517760_589ce1e83a1df80f7b3b569c_005 abc_0051_00517764_589ce2c308d6170f75ad0302_000 abc_0051_00517778_112c441b377c9f99f5849ff2_000 abc_0051_00517786_c8745c064381ae55fad72001_004 abc_0051_00517796_836c045df81452a9654b13ef_001 abc_0051_00517802_486ace821ef499b593f72d64_000 abc_0051_00517949_aabc24344bb566626be4bd02_000 abc_0051_00517983_0bf71c8f7dd022fab51b8c00_000 abc_0051_00518275_589d18eadd8b280f7ba860bb_000






do

    sbatch sharp_features/contrib/ec_net/run_ecnet_in_docker_images_real.sh -i ${INPUT_FOLDER}/${filename}.hdf5 -o ${INPUT_FOLDER}/${filename} -l ${LABEL} -v ${VARLEN}
    sbatch sharp_features/contrib/sharpness_fields/run_sharpness_fields_in_docker_images_real.sh -i ${INPUT_FOLDER}/${filename}.hdf5 -o ${INPUT_FOLDER}/${filename} -l ${LABEL} -v ${VARLEN}
    sbatch sharp_features/contrib/CGAL/voronoi_1/run_voronoi_in_docker_images_real.sh -i ${INPUT_FOLDER}/${filename}.hdf5 -o ${INPUT_FOLDER}/${filename}/v_result.hdf5 -R ${V_OFFSET_RADIUS} -r ${V_CONV_RADIUS} -t ${V_THRESHOLD} -j 10 -v ${VARLEN}
#     sbatch scripts/merge_images.sh -i ${INPUT_FOLDER}/${filename}/ec_net_results/ -f ${INPUT_FOLDER}/${filename}.hdf5 -d ${LABEL} -o ${INPUT_FOLDER}/${filename}/ec_net_results/ec_result.hdf5
    
done