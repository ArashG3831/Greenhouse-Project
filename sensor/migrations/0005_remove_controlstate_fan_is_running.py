# Generated by Django 5.1.6 on 2025-03-25 00:54

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('sensor', '0004_controlstate_fan_is_running_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='controlstate',
            name='fan_is_running',
        ),
    ]
