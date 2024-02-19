from setuptools import setup

setup(
    name='Kelpie',
    version='1.0',
    description='',
    author='',
    author_email='',
    packages=['Kelpie',
                'Kelpie.explanation_builders',
                'Kelpie.link_prediction',
                'Kelpie.link_prediction.evaluation',
                'Kelpie.link_prediction.models',
                'Kelpie.link_prediction.optimization',
                'Kelpie.link_prediction.regularization',
              'Kelpie.prefilters', 'Kelpie.relevance_engines', 'Kelpie.scripts', 'Kelpie.scripts.complex', 'Kelpie.scripts.conve', 'Kelpie.scripts.transe'],
)
